# ilpc的代码
import pickle
import numpy as np
import os
import faiss
import scipy as sp
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import t
from sklearn import metrics
from test_arguments import parse_option


# 加载plk文件
def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


# 输出图像的数据和标签
def loadDataSet(dsname):
    _datasetFeaturesFiles = {"miniImagenet": "../checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk",
                             "CUB": "../checkpoints/CUB/WideResNet28_10_S2M2_R/last/output.plk",
                             "cifar": "../checkpoints/cifar/WideResNet28_10_S2M2_R/last/output.plk",
                             "tieredImagenet": "../checkpoints/tieredImagenet/WideResNet28_10_S2M2_R/last/output.plk"}

    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName
    dsName = dsname
    # Loading data from files on computer
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])


    global _min_examples
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    global data
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    global labels
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
        [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))


# 生成随机状态
def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])
    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']+cfg['queries']]
    return dataset


# 调用随机状态
def setRandomStates(cfg):
    global _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(42)
        global _randStates
        _randStates = []
        global _maxRuns
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


# 生成符合随机状态的图像数据
def GenerateRunSet(start=None, end=None, cfg=None):
    global _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))
    global dataset
    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)
    return dataset


# 缩放每张图片的数据
def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


# 对数据进行中心化
def centerDatas(datas):
    # centre of mass of all data support + querries
    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
    datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
    return datas


# 生成及更新伪标签
def update_plabels(opt, support, support_ys, query):
    max_iter = 20
    no_classes = support_ys.max() + 1
    X = np.concatenate((support, query), axis=0).copy(order='C')
    k = opt.K
    alpha = opt.alpha
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]] = support_ys
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]

    # kNN search for the graph
    d = X.shape[1]

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    D, I = index.search(X, k + 1)
    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T

    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))),
                             shape=(N, N))  # 公式1

    W = W + W.T
    W = W - sp.sparse.diags(W.diagonal())

    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D  # 公式2


    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, no_classes))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(no_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0
    # --------try to filter-----------
    z_amax = -1 * np.amax(Z, 1)[support_ys.shape[0]:]

    # -----------trying filtering--------
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)

    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1, 1)

    p_labels[labeled_idx] = labels[labeled_idx]
    return p_labels[support.shape[0]:], probs_l1[support.shape[
                                                     0]:], z_amax  # p_probs #weights[support.shape[0]:]


# 计算最好的标签
def compute_optimal_transport(opt, M, epsilon=1e-6):  # 1
    r = torch.ones(1, M.shape[0])
    # r = r * weights
    # c = torch.ones(1, M.shape[1]) * int(M.shape[0]/M.shape[1])
    c = torch.FloatTensor(opt.no_samples)
    idx = np.where(c.detach().cpu().numpy() <= 0)
    if opt.unbalanced == True:
        c = torch.FloatTensor(opt.no_samples)
        idx = np.where(c.detach().cpu().numpy() <= 0)
        if len(idx[0]) > 0:
            M[:, idx[0]] = torch.zeros(M.shape[0], 1)
    M = M.cuda()
    r = r.cuda()
    c = c.cuda()
    M = torch.unsqueeze(M, dim=0)
    n_runs, n, m = M.shape
    P = M

    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    for i in range(opt.sinkhorn_iter):
        P = torch.pow(P, opt.T)
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if len(idx[0]) > 0:
                P[P != P] = 0
            if iters == maxiters:
                break
            iters = iters + 1
    P = torch.squeeze(P).detach().cpu().numpy()
    best_per_class = np.argmax(P, 0)
    if M.shape[1] == 1:
        P = np.expand_dims(P, axis=0)
    labels = np.argmax(P, 1)
    return P, labels, best_per_class


# 返回全连接层
def weight_imprinting(X, Y, model):
    no_classes = Y.max() + 1
    imprinted = torch.zeros(no_classes, X.shape[1])
    for i in range(no_classes):
        idx = np.where(Y == i)
        tmp = torch.mean(X[idx], dim=0)
        tmp = tmp / tmp.norm(p=2)
        imprinted[i, :] = tmp
    model.weight.data = imprinted
    return model


# 计算损失函数
def label_denoising(opt, support, support_ys, query, query_ys_pred):
    all_embeddings = np.concatenate((support, query), axis=0)
    input_size = all_embeddings.shape[1]
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    start_lr = 0.1
    end_lr = 0.1
    cycle = 50
    step_size_lr = (start_lr - end_lr) / cycle
    # print(input_size, output_size.item())
    lambda1 = lambda x: start_lr - (x % cycle) * step_size_lr
    o2u = nn.Linear(input_size, output_size.item())
    o2u = weight_imprinting(torch.Tensor(all_embeddings[:support_ys.shape[0]]), support_ys, o2u)

    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_statistics = torch.zeros(all_ys.shape, requires_grad=True)
    lr_progression = []
    for epoch in range(1000):
        output = o2u(X)
        optimizer.zero_grad()
        loss_each = criterion(output, Y)
        loss_each = loss_each  # * weights
        loss_all = torch.mean(loss_each)
        loss_all.backward()
        loss_statistics = loss_statistics + loss_each / (opt.denoising_iterations)
        optimizer.step()
        scheduler_lr.step()
        lr_progression.append(optimizer.param_groups[0]['lr'])
    return loss_statistics, lr_progression


# 对损失值排序，输出伪标签中与5个class中值最小的3个,共15个
def rank_per_class(no_cls, rank, ys_pred, no_keep):
    list_indices = []
    list_ys = []
    for i in range(no_cls):
        cur_idx = np.where(ys_pred == i)
        y = np.ones((no_cls,)) * i
        class_rank = rank[cur_idx]
        class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
        class_rank_sorted[class_rank_sorted > no_keep] = 0
        indices = np.nonzero(class_rank_sorted)
        list_indices.append(cur_idx[0][indices[0]])
        list_ys.append(y)
    idxs = np.concatenate(list_indices, axis=0)
    ys = np.concatenate(list_ys, axis=0)
    return idxs, ys


# 仍然存在的伪标签
def remaining_labels(opt, selected_samples):  # 4
    for i in range(len(opt.no_samples)):
        occurrences = np.count_nonzero(selected_samples == i)
        opt.no_samples[i] = opt.no_samples[i] - occurrences


# 返回准确率的平均值和误差
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


# 获取伪标签，并迭代更新
def iter_balanced_trans(opt, support_features, support_ys, query_features, query_ys, labelled_samples):
    query_ys_updated = query_ys
    total_f = support_ys.shape[0] + query_ys.shape[0]
    iterations = int(query_ys.shape[0])

    for j in range(iterations):
        query_ys_pred, probs, weights = update_plabels(opt, support_features, support_ys, query_features)

        P, query_ys_pred, indices = compute_optimal_transport(opt, torch.Tensor(probs))

        loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features,
                                             query_ys_pred)
        un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()  # np.amax(P, 1)
        rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')
        # rank = sp.stats.rankdata(weights, method='ordinal')

        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, opt.best_samples)
        if len(indices) < 5:
            break
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)

        # if opt.unbalanced:
        remaining_labels(opt, pseudo_ys)
        if support_features.shape[0] == total_f:
            break

    # 输出选择伪标签的结果
    #     support_ys_temp = np.concatenate((support_ys, query_ys_pred), axis=0)
    #     query_ys_temp = np.concatenate((query_ys, query_ys_updated), axis=0)
    #     query_ys_pred_temp = support_ys_temp[labelled_samples:]
    #     query_ys_temp = query_ys_temp[query_ys_pred_temp.shape[0]:]
    #     print(metrics.accuracy_score(query_ys_temp, query_ys_pred_temp))
    # print()

    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    support_features = np.concatenate((support_features, query_features), axis=0)
    query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    query_ys_pred = support_ys[labelled_samples:]
    query_ys = query_ys[query_ys_pred.shape[0]:]
    return query_ys, query_ys_pred


# 计算1000个随机状态下的准确率和误差
def trans_ilpc(opt, X, Y, labelled_samples):
    acc = []
    # for i in range(10):
    for i in range(X.shape[0]):
        if i % 400 == 0:
            print("ilpc: ", i)
        support_features, query_features = X[i, :labelled_samples], X[i,
                                                                    labelled_samples:]  # X_pca[:labelled_samples], X_pca[labelled_samples:]
        support_ys, query_ys = Y[i, :labelled_samples], Y[i, labelled_samples:]
        labelled_samples = support_features.shape[0]
        if params.unbalanced == True:
            query_features, query_ys, opt.no_samples = unbalancing(opt, query_features, query_ys)
        else:
            opt.no_samples = np.array(np.repeat(float(query_ys.shape[0] / opt.n_ways), opt.n_ways))
        query_ys, query_ys_pred = iter_balanced_trans(opt, support_features, support_ys, query_features,
                                                      query_ys,
                                                      labelled_samples)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        # break  # *
    return mean_confidence_interval(acc)


if __name__ == '__main__':
    params = parse_option()
    # ---- data loading
    n_shot = params.n_shots
    n_ways = params.n_ways
    n_unlabelled = params.n_unlabelled
    n_queries = params.n_queries
    n_runs = 1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    dataset = params.dataset
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}

    loadDataSet(dataset)
    _rsCfg = None
    _cacheDir = "./cache"
    _maxRuns = 1000
    setRandomStates(cfg)
    ndatas = GenerateRunSet(cfg=cfg)
    dataset = params.dataset
    print(ndatas.shape)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    print(params.unbalanced)

    # Power transform
    beta = 0.5
    # ------------------------------------PT-MAP-----------------------------------------------
    nve_idx = np.where(ndatas.cpu().detach().numpy() < 0)
    ndatas[nve_idx] *= -1
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    ndatas[nve_idx] *= -1
    # ------------------------------------------------------------------------------------------
    print(ndatas.type())
    n_nfeat = ndatas.size(2)
    ndatas_icp = ndatas

    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas)
    print("size of the datas...", ndatas.size())

    acc_mine, acc_std = trans_ilpc(params, ndatas, labels, n_lsamples)
    print('DATASET: {}, final accuracy ilpc: {:0.2f} +- {:0.2f}, shots: {}, queries: {}'.format(dataset,
                                                                                                acc_mine * 100,
                                                                                                acc_std * 100,
                                                                                                n_shot, n_queries))
