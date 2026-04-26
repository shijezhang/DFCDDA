import numpy as np
import torch as th
import torch.nn as nn
from parse_args import args
from model import DFCDDA
from util.evaluate import get_metrics
from data_preprocessing import process_data, dgl_heterograph
from diffusion.resample import UniformSampler, LossAwareSampler
from contrastive_learning import similarity_contrastive, inter_contrastive
from util.script_util import (
    diffusion_defaults,
    create_diffusion,
    args_to_dict,
)


def train(model, args, d_data, drdipr_graph, dr_m, di_m, pr_m,
          drdr_matrix, didi_matrix, X_train, X_test, Y_train, Y_test):
    drf = th.FloatTensor(d_data['drf']).to(args.device)
    dip = th.FloatTensor(d_data['dip']).to(args.device)
    dr_raw = th.tensor(d_data['drf'], dtype=th.float32).to(args.device)
    di_raw = th.tensor(d_data['dip'], dtype=th.float32).to(args.device)
    drf_knn = d_data['drf_knn'].to(args.device)
    dip_knn = d_data['dip_knn'].to(args.device)
    drg_knn = d_data['drg_knn'].to(args.device)
    dig_knn = d_data['dig_knn'].to(args.device)

    print("Start Training ...")
    best_metric = -float("inf")

    rel_loss = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    schedule_sampler_x = UniformSampler(diffusion_x)
    schedule_sampler_y = UniformSampler(diffusion_y)

    for iter_idx in range(1, args.max_epochs):
        model.train()
        train_score, dr_hgt, dr_sim, di_hgt, di_sim, dr_high, di_high = \
            model(drf_knn, drf, drg_knn,
                  dip_knn, dip, dig_knn,
                  drdipr_graph, dr_m, di_m, pr_m, X_train)

        t_x, weights_x = schedule_sampler_x.sample(dr_raw.shape[0], dr_raw.device)
        model_kwargs = {'y': dr_high.detach()}
        loss_diffx = diffusion_x.training_losses(model.unet_x, (dr_raw - 0.5) * 2, t_x, model_kwargs)
        loss_drug = (loss_diffx["loss"] * weights_x).mean()

        t_y, weights_y = schedule_sampler_y.sample(di_raw.shape[0], di_raw.device)
        model_kwargs = {'y': di_high.detach()}
        loss_diffy = diffusion_y.training_losses(model.unet_y, (di_raw - 0.5) * 2, t_y, model_kwargs)
        if isinstance(schedule_sampler_y, LossAwareSampler):
            schedule_sampler_y.update_with_local_losses(
                t_y, loss_diffy["loss"].detach()
            )
        loss_dis = (loss_diffy["loss"] * weights_y).mean()
        loss_diff = loss_drug + loss_dis

        intra_contrastive_loss = similarity_contrastive(drdr_matrix, didi_matrix, dr_sim, di_sim, args)
        inter_contrastive_loss = inter_contrastive(drdr_matrix, didi_matrix, dr_sim, di_sim, dr_hgt,
                                                   di_hgt, args)
        loss_contra = inter_contrastive_loss + intra_contrastive_loss

        train_score = train_score.squeeze(-1)
        loss_main = rel_loss(train_score, Y_train)

        loss = loss_main + args.alpha * loss_diff + args.beta * loss_contra

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with th.no_grad():
            model.eval()
            test_score, _, _, _, _, _, _ = model(drf_knn, drf, drg_knn, dip_knn, dip, dig_knn,
                                                 drdipr_graph, dr_m, di_m, pr_m, X_test)

        auroc, aupr, accuracy, precision, recall, f1, mcc = get_metrics(test_score, Y_test)

        mix_factor = auroc + aupr + mcc
        if mix_factor > best_metric:
            best_metric = mix_factor

        logging_str = "Iter={}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f}, " \
                      "ACC={:.4F}, Precision={:.4F}, Recall={:.4F}, F1={:.4F}, MCC={:.4F}".format(
            iter_idx, loss.item(), auroc, aupr, accuracy, precision, recall, f1, mcc)

        if iter_idx % 100 == 0:
            print(logging_str)


if __name__ == '__main__':

    if args.dataset in ["B-dataset"]:
        args.lr = 1e-4

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    diffusion_x = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    diffusion_y = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )

    d_data, drdr_graph, didi_graph = process_data(args)
    drdr_graph = drdr_graph.to(args.device)
    didi_graph = didi_graph.to(args.device)
    args.dr_num = d_data['drug_number']
    args.di_num = d_data['disease_number']

    for fold_i in range(args.K_fold):
        print("============" + str(args.dataset) + "_fold_" + str(fold_i + 1) + "============")

        positive_num = int(np.sum(d_data['Y_train'][fold_i] == 1))
        np_X_train_fold_i_positive = d_data['X_train'][fold_i][:int(args.dataset_percent * positive_num)]

        heterograph = dgl_heterograph(d_data, np_X_train_fold_i_positive)
        heterograph = heterograph.to(args.device)
        meta_g = heterograph.metagraph()

        dr_m = th.FloatTensor(d_data['drugfeature']).to(args.device)
        di_m = th.FloatTensor(d_data['diseasefeature']).to(args.device)
        pr_m = th.FloatTensor(d_data['proteinfeature']).to(args.device)
        drdr_matrix_bool = d_data["drdr_matrix"]
        didi_matrix_bool = d_data["didi_matrix"]

        fold_data = d_data['fold_data']
        graph_data = fold_data[fold_i]

        X_train = graph_data['train'][1].int().to(args.device)
        X_test = graph_data['test'][1].int().to(args.device)
        Y_train = graph_data['train'][2].to(args.device)
        Y_test = graph_data['test'][2]

        model = DFCDDA(args, meta_g)
        model = model.to(args.device)

        train(model, args, d_data, heterograph, dr_m, di_m, pr_m,
              drdr_matrix_bool, didi_matrix_bool, X_train, X_test, Y_train, Y_test)
