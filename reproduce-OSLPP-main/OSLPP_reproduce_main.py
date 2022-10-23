import math
from sklearn.decomposition import PCA
import scipy
import numpy as np
import scipy.io
import scipy.linalg
from dataclasses import dataclass


def get_l2_norm(features: np.ndarray):  # features: np.ndarray指定参数类型为np.ndarray
    # 求得每张图片第1维的数据的L2范数(平方和开根号)
    return np.sqrt(np.square(features).sum(axis=1)).reshape((-1, 1))


def get_l2_normalized(features: np.ndarray):  # l2标准化
    # 每张图的2048维数据都除以L2范数
    return features / get_l2_norm(features)


def get_dist(f, features):  # 第一轮大循环中，此循环执行2427次数
    # 第一轮循环中，f为图片对应的128维信息 (128,)；features(centroids) (25, 128) 为128张图片对应25个标签的1个中心值
    # 第二轮循环以后，f未标记标签信息，features已选择标签信息
    return get_l2_norm(f - features)  # f - features (25, 128) 第一轮表示f广播成(25, 128)之后与features相减
    # 第二轮循环以后，表示求(此未标记标签信息广播后-已选择标签信息)的L2范数
    # 最终求得每张图片第1维的128维数据的L2范数(平方和开根号)


def do_l2_normalization(feats_S, feats_T):
    # 1.feats_S为0-24标签的的Clipart数据集(1675, 2048)；feats_T为大于25标签全部设置为25的Clipart数据集(2427, 2048)
    # 2.feats_S为0-24标签的的Clipart数据集(1675, 512)；feats_T为大于25标签全部设置为25的Clipart数据集(2427, 512)
    # 3.循环中feats_S为0-24标签的的Clipart数据集(1675, 128)；feats_T为大于25标签全部设置为25的Clipart数据集(2427, 128)
    feats_S, feats_T = get_l2_normalized(feats_S), get_l2_normalized(feats_T)  # 对feats_S, feats_T进行L2标准化
    # 标准化后的L2范数为1(易证)
    assert np.abs(get_l2_norm(feats_S) - 1.).max() < 1e-5
    assert np.abs(get_l2_norm(feats_T) - 1.).max() < 1e-5
    return feats_S, feats_T
    # 返回标准化后的数据


@dataclass
class Params:
    pca_dim: int  # = 512
    proj_dim: int  # = 128
    T: int  # = 10
    n_r: int  # = 1200
    dataset: str  # = 'OfficeHome'
    source: str  # = 'art'
    target: str  # = 'clipart'
    num_src_classes: int  # = 25
    num_total_classes: int  # = 65


def main(params: Params):
    # pca_dim=512, proj_dim=128, T=10, n_r=1200, dataset='OfficeHome',
    # source='clipart', target='art', num_src_classes=25, num_total_classes=65

    def create_datasets(source, target, num_src_classes, num_total_classes):
        # source='clipart';target='art';num_src_classes=25;num_total_classes=65
        def _load_tensors(domain):  # 作用：domain为输入的数据集名称，输出数据和对应的标签
            # domain(source)='clipart'
            mapping = {  # 映射
                'art': 'Art',
                'clipart': 'Clipart',
                'product': 'Product',
                'real_world': 'RealWorld'
            }
            mat = scipy.io.loadmat(f'mats/OfficeHome-{mapping[domain]}-resnet50-noft.mat')
            # 1.加载OfficeHome-Clipart-resnet50-noft.mat  2.加载OfficeHome-Art-resnet50-noft.mat
            # mat为一个字典
            features, labels = mat['resnet50_features'], mat['labels']
            # features和labelsmat中关键字resnet50_features和labels对应的值；labels为0,0...,64,64
            # 1.features为4365张图片x2048维x1x1，labels为1个标签x4365张图片  2.features为2427张图片x2048维x1x1，labels为1x2427个标签
            features, labels = features[:, :, 0, 0], labels[0]  # 降维：1.(4365, 2048) (4365,) 2.(2427, 2048) (2427,)
            assert len(features) == len(labels)  # len(features) == len(labels)
            # features, labels = torch.tensor(features), torch.tensor(labels)
            # features = torch.load(f'./data_handling/features/OH_{domain}_features.pt')
            # labels = torch.load(f'./data_handling/features/OH_{domain}_labels.pt')
            return features, labels  # 返回数据和标签
        src_features, src_labels = _load_tensors(source)
        # src_features, src_labels表示Clipart数据集对应的数据(4365, 2048)和标签(4365,)

        idxs = src_labels < num_src_classes  # num_src_classes=25
        # idxs为(4365,)的bool类型，小于25为true，大于为false

        src_features, src_labels = src_features[idxs], src_labels[idxs]  # src_features(1675, 2048), src_labels(1675,)
        # 取小于25的标签和对应的数据；0-24每个类有67张图，一共1675个标签和数据

        tgt_features, tgt_labels = _load_tensors(target)
        # tgt_features, tgt_labels表示Art数据集对应的数据(2427, 2048)和标签(2427,)

        idxs = tgt_labels < num_total_classes  # num_total_classes=65
        # idxs为(2427,)的bool类型，小于65为true，大于为false(全为true)

        tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]  # tgt_features(2427, 2048), tgt_labels(2427,)
        # 0-64一共65类，共有2427个标签和数据

        tgt_labels[tgt_labels >= num_src_classes] = num_src_classes  # num_src_classes=25, tgt_labels(2427,)
        # tgt_labels的2427个标签中大于或等于25的标签，全部设置为25

        assert (np.unique(src_labels) == np.arange(0, num_src_classes)).all()  # src_labels为0-24
        # (np.unique(src_labels) == np.arange(0, num_src_classes))为25个全为true组成的numpy数组
        # np.unique去除数组中的重复数字，排序后输出； all()函数判断整个数组中的元素的值是否全部满足条件，满足返回True，否则返回False

        assert (np.unique(tgt_labels) == np.arange(0, num_src_classes + 1)).all()  # tgt_labels为0-25
        assert len(src_features) == len(src_labels)  # 数据个数和标签个数相等
        assert len(tgt_features) == len(tgt_labels)

        return (src_features, src_labels), (tgt_features, tgt_labels)
        # 返回选中0-24标签的的Clipart数据集(1675, 2048)和标签(1675,)；大于25标签全部设置为25的Art数据集(2427, 2048)和标签(2427,)
    (feats_S, lbls_S), (feats_T, lbls_T) = create_datasets(params.source, params.target,
                                                           params.num_src_classes, params.num_total_classes)
    # (feats_S, lbls_S)为0-24标签的的Clipart数据集的数据(1675, 2048)和标签(1675,)
    # (feats_T, lbls_T)为大于25标签全部设置为25的Art数据集的数据(2427, 2048)和标签(2427,)
    # 创建数据集(图片数据+标签)  这两个数据集的样本数都不均匀

    # for c in np.unique(lbls_T):
    #     if c == 25:
    #         idxs = np.where(lbls_T == c)[0]
    #         print(len(idxs))  # 这个标签的个数

    # l2 normalization and pca  L2标准化和PCA
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    # feats_S为0-24标签的的Clipart数据集(1675, 2048)；feats_T为大于25标签全部设置为25的Art数据集(2427, 2048)

    def do_pca(feats_S, feats_T, pca_dim):  # pca_dim=512
        feats = np.concatenate([feats_S, feats_T], axis=0)  # 合并两个数据集后的数据(4102, 2048)

        def get_PCA(features, dim):  # features(feats)为合并两个数据集后的数据(4102, 2048)；dim(pca_dim)=512
            result = PCA(n_components=dim).fit_transform(features)  # 作用：4102张图片的2048维数据降维到512维数据
            # n_components：指定希望PCA降维后的特征维度数目；fit_transform(features)：用features来训练PCA模型，同时返回降维后的数据。

            assert len(features) == len(result)
            return result  # 返回降维到512维数据后的结果(4102, 512)
        feats = get_PCA(feats, pca_dim)  # pca_dim=512；feats(4102, 512) numpy.ndarray
        # 使用PCA降维
        feats_S, feats_T = feats[:len(feats_S)], feats[len(feats_S):]  # 再将两个数据集的数据分开
        return feats_S, feats_T  # 作用：降维
    feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
    # 降维到(1675, 512) (2427, 512)

    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    # 返回标准化后的数据

    # initial 初始设置
    feats_all = np.concatenate((feats_S, feats_T), axis=0)  # 合并两个数据集的数据(4102, 512)
    pseudo_labels = -np.ones_like(lbls_T)  # pseudo_labels为Art数据集的标签(2427,), 全部用-1填充
    rejected = np.zeros_like(pseudo_labels)  # rejected为全0的numpy数组，shape(2427,)
    # iterations 迭代
    for t in range(1, params.T + 1):  # T=10  t=range(1, 11)
        # for c in np.unique(pseudo_labels):
        #     if c <= 25:
        #         idxs = np.where(pseudo_labels == c)[0]
        #         print(c, len(idxs))  # 这个标签的个数
        # print("all")
        def get_projection_matrix(features, labels, proj_dim):  # 得到投影矩阵
            # features(feats_all)合并两个数据集的数据(4102, 512)；labels(np.concatenate((lbls_S, pseudo_labels))中，
            # lbls_S为0-24标签的的Clipart数据集标签(1675,),pseudo_labels为Art数据集的标签(2427,),用-1填充,连接得到(4102,);proj_dim=128

            N, d = features.shape  # (4102, 512)
            X = features.transpose()  # numpy.transpose交换坐标轴 4102张图的512维数据交换坐标轴(512, 4102)

            def get_W(labels, ):  # labels (4102,)的numpy数组，前1675为0-24，后2427为-1
                W = (labels.reshape(-1, 1) == labels).astype(np.int)  # labels.reshape(-1, 1) (4102, 1)
                # np.astype就是转换numpy数组的数据类型;labels.reshape(-1, 1):[[ 0] [ 0] [ 0] ... [-1] [-1] [-1]]
                # labels:[ 0  0  0 ... -1 -1 -1];numpy数组不同维度的变量进行相等判断时，高维变量每个维度元素为1，
                # 低维元素会依次与低维变量的每个元素进行相等判断
                # W (4102, 4102)；第一次循环：对角线的子矩阵按照0-24，-1依次为全1，其余为0
                negative_one_idxs = np.where(labels == -1)[0]  # negative_one_idxs表示labels中值为-1的元素的下标，初始shape为2427，逐渐减少为空
                W[:, negative_one_idxs] = 0  # negative_one_idxs对应行列全部元素变成0
                W[negative_one_idxs, :] = 0
                return W
            W = get_W(labels)  # labels (4102,)的numpy数组，前1675为0-24，后2427为-1
            # W (4102, 4102)，行列都对应0-24标签的矩阵为全1，其余为全0

            def get_D(W):
                return np.eye(len(W), dtype=np.int) * W.sum(axis=1)  # np.eye(len(W), dtype=np.int) (4102, 4102)的单位矩阵
                # W.sum(axis=1)表示这一列0-24元素的个数
            D = get_D(W)  # 返回(4102, 4102)的对角矩阵，对角线元素表示标签(0-24)的个数
            L = D - W  # 对角阵减去原矩阵(仅与0-24各自多少个有关)

            def fix_numerical_assymetry(M):  # M (512, 512)
                return (M + M.transpose()) * 0.5  # 两个维度的元素互换，对应位置与原值相加取平均值
            A = fix_numerical_assymetry(np.matmul(np.matmul(X, D), X.transpose()))
            # X(512, 4102)512维数据x4102张图片, D(4102, 4102)对角矩阵
            # np.matmul矩阵乘法
            # np.matmul(X, D) (512, 4102) 给每个数据乘上对应标签的数目, X.transpose() (4102, 512) 4102张图片x512维数据
            # np.matmul(np.matmul(X, D), X.transpose()) (512, 512) 不同维度的4102张图片对应数据，对应相乘再求和
            # A (512, 512) 为最终求得的值为两个维度互换，再取平均值，根据全部512维数据，考虑各个标签的数目计算得到的信息
            B = fix_numerical_assymetry(np.matmul(np.matmul(X, L), X.transpose()) + np.eye(d))  # d=512
            # X(512, 4102)512维数据x4102张图片, L(4102, 4102)对角矩阵D减去W的矩阵
            # np.matmul(X, L) (512, 4102) 相同标签(0-24)的图片对应维度方阵与全1方阵相乘(不同图片对应维度求和), X.transpose() (4102, 512)
            # np.matmul(np.matmul(X, L), X.transpose()) (512, 512) 不同维度的4102张图片对应数据，对应相乘再求和，然后加上单位矩阵
            # B (512, 512) 为最终求得的值为两个维度互换，再取平均值，根据全部512维数据，考虑各个标签的特点计算得到的信息
            assert (A.transpose() == A).all() and (B.transpose() == B).all()  # A和B都为对称矩阵

            w, v = scipy.linalg.eigh(A, B)  # A实对称矩阵, B实对称正定矩阵；A和B都为(512,512)矩阵；
            # 查找特征值数组 w 和可选的特征向量数组 v 的数组 A
            # B为正定矩阵，使得对于每个特征值 λ (w 的i-th 条目)及其特征向量 vi (i-th v 的列)满足:
            # A @ vi = λ * B @ vi; vi.conj().T @ A @ vi = λ; vi.conj().T @ B @ vi = 1  conj()共轭复数
            # 返回：w：(N,) 数组N(1<=N<=M)个选择的特征值，按升序排列，每个都根据其多重性重复。v： (M, N) ndarray
            # w (512,)表示A的特征值；v(512, 512)为对应于特征值的归一化特征向量

            # print(np.allclose(np.matmul(A, v[:,1]) - w[1] * np.matmul(B, v[:,1]), np.zeros((512, 512))))
            # print(np.allclose(np.matmul(A, v) - w * np.matmul(B, v), np.zeros((512, 512))))
            # print(np.allclose(v[:,1].conj().T @ A @ v[:,1] - w[1], np.zeros((512, 512))))
            # print(np.allclose(v[:,1].conj().T @ B @ v[:,1] - 1, np.zeros((512, 512))))

            assert w[0] < w[-1]  # 特征值不全相等

            w, v = w[-proj_dim:], v[:, -proj_dim:]  # proj_dim=128  取后128个(特征值大的)
            # 取128个最大的特征值，以及对应的特征向量
            assert np.abs(np.matmul(A, v) - w * np.matmul(B, v)).max() < 1e-5
            w = np.flip(w)  # 顺序颠倒
            v = np.flip(v, axis=1)  # axis=1：左右翻转，意味着把列看成整体，列的顺序发生颠倒，每一列的元素不发生改变

            for i in range(v.shape[1]):  # v.shape[1]=128 第0维小于0的取正数
                if v[0, i] < 0:
                    v[:, i] *= -1
            return v  # v为数据处理后的实对称矩阵的特征向量经过一系列处理后的最大128特征值对应的特征向量
        P = get_projection_matrix(feats_all, np.concatenate((lbls_S, pseudo_labels), axis=0), params.proj_dim)
        # feats_all合并两个数据集的数据(4102, 512)；lbls_S为0-24标签的的Clipart数据集的标签(1675,)
        # pseudo_labels为Art数据集的标签(2427,), 全部用-1填充,连接在一起得到(4102,)；proj_dim=128
        # P为数据处理后的实对称矩阵的特征向量经过一系列处理后的最大128特征值对应的特征向量(512, 128)

        def project_features(P, features):  # 投影特征
            # P: pca_dim x proj_dim
            # features: N x pca_dim
            # result: N x proj_dim
            return np.matmul(P.transpose(), features.transpose()).transpose()
        # P为数据处理后的实对称矩阵的特征向量经过一系列处理后的最大128特征值对应的特征向量(512, 128)，P.transpose() (128, 512)
        # features(feats_all)合并两个数据集的数据(图片数, 512)，features.transpose() (512, 图片数)
        # 计算两个的矩阵乘积(128, 图片数)，返回转置(图片数, 128)
        # 作用：将128个最大特征值的特征向量的特征映射到全部图片上

        proj_S, proj_T = project_features(P, feats_S), project_features(P, feats_T)
        # P为数据处理后的实对称矩阵的特征向量经过一系列处理后的最大128特征值对应的特征向量(512, 128)
        # feats_S、feats_T降维到(1675, 512) (2427, 512)
        # proj_S, proj_T返回(1675, 128) (2427, 128)  将128个最大特征值的特征向量的特征映射到全部图片上

        def center_and_l2_normalize(zs_S, zs_T):
            # zs_S, zs_T返回(1675, 128) (2427, 128)  将128个最大特征值的特征向量的特征映射到全部图片上
            # center
            zs_mean = np.concatenate((zs_S, zs_T), axis=0).mean(axis=0).reshape((1, -1))  # 连接+求平均值+增加第0维 (1,128)的平均值数据
            zs_S = zs_S - zs_mean  # 每个数据减去平均值
            zs_T = zs_T - zs_mean
            # l2 normalize l2标准化
            zs_S, zs_T = do_l2_normalization(zs_S, zs_T)  # l2标准化 zs_S,zs_T  (1675, 128),(2427, 128)
            return zs_S, zs_T
        proj_S, proj_T = center_and_l2_normalize(proj_S, proj_T)
        # 输入proj_S, proj_T返回(1675, 128) (2427, 128)  将128个最大特征值的特征向量的特征映射到全部图片上
        # 中心对称和l2标准化 输出proj_S, proj_T (1675, 128),(2427, 128)

        def get_closed_set_pseudo_labels(features_S, labels_S, features_T):
            # features_S(proj_S), features_T(proj_T)为(1675, 128),(2427, 128)；labels_S(lbls_S)为0-24标签的的Clipart数据集的标签(1675,)
            def get_centroids(features, labels):
                # features(features_S) (1675, 128);labels(labels_S)为0-24标签的的Clipart数据集的标签(1675,)
                centroids = np.stack([features[labels == c].mean(axis=0) for c in np.unique(labels)], axis=0)
                # np.unique 唯一化，删除重复元素 c为0-24
                # [features[labels == c].mean(axis=0)标签为0-24时的选出的128张图片对应标签c的不同维度的平均值  (1,128)
                # np.stack([arrays1,array2,array3],axis=0)。堆叠。整个函数的输出为一个新数组。centroids (25, 128)
                centroids = get_l2_normalized(centroids)  # 将centroids进行l2标准化
                return centroids  # (25, 128)
            centroids = get_centroids(features_S, labels_S)  # features_S(1675, 128),labels_S为0-24标签的的Clipart数据集的标签(1675,)
            # centroids (25, 128) 128张图片，每一张对应一个维度求一个中心值
            dists = np.stack([get_dist(f, centroids)[:, 0] for f in features_T], axis=0)
            # features_T为(2427, 128) f分别为2427图片对应的128维信息 (128,)；centroids (25, 128) 为128张图片对应25个标签的1个中心值
            # [get_dist(f, centroids)[:, 0] for f in features_T]最终求得2427张图片的图片第1维的128维数据的L2范数(平方和开根号)
            # dists为(2427, 25) 2427张图片在25个标签上的值(都为1-2之间的数)
            pseudo_labels = np.argmin(dists, axis=1)  # np.argmin给出axis=1方向最小值的下标
            # pseudo_labels (2427,) 2427张图片在25个标签中的最小值
            pseudo_probs = np.exp(-dists[np.arange(len(dists)), pseudo_labels]) / np.exp(-dists).sum(axis=1)
            # dists为(2427, 25) 2427张图片在25个标签上的值，len(dists)为2427；pseudo_labels (2427,) 2427张图片在25个标签中的最小值
            # dists横坐标对应np.arange(len(dists)) 0-2426, 纵坐标为对应位置的pseudo_labels的值，得到2427个值
            # np.exp(x)   e的x幂次方
            # np.exp(-dists[np.arange(len(dists)), pseudo_labels]) (2427,)为0~2426的e的-x幂次方(x为pseudo_labels中对应元素)
            # np.exp(-dists)为对应于(2427, 25)的 e的-x幂次方，x为dist中元素
            # np.exp(-dists).sum(axis=1) (2427,)对np.exp(-dists)第1维25个数求和
            # pseudo_probs (2427,)为以上二者对应相除的结果(伪标签数字/数据)
            return pseudo_labels, pseudo_probs  # 返回标签值和对应的值

        pseudo_labels, pseudo_probs = get_closed_set_pseudo_labels(proj_S, lbls_S, proj_T)
        # proj_S, proj_T (1675, 128),(2427, 128) lbls_S为0-24标签的的Clipart数据集的标签(1675,)
        # pseudo_labels, pseudo_probs (2427,),(2427,) 返回标签值和对应的值

        def select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, T):
            # pseudo_labels, pseudo_probs (2427,),(2427,) 对应伪标签值和数据值;t为迭代的轮次1-10;T=10
            if t >= T: t = T - 1  # t=10时，减1为9
            selected = np.zeros_like(pseudo_labels)  # (2427,)的全0
            for c in np.unique(pseudo_labels):  # np.unique(pseudo_labels)为0-24
                idxs = np.where(pseudo_labels == c)[0]  # idxs表示伪标签为c的位置
                Nc = len(idxs)  # 这个标签的个数
                if Nc > 0:
                    class_probs = pseudo_probs[idxs]  # 全部伪标签c对应的值
                    class_probs = np.sort(class_probs)  # class_probs中的值从小到大排序
                    threshold = class_probs[math.floor(Nc * (1 - t / (T - 1)))]
                    # Nc * (1 - t / (T - 1)) = Nc * (1 - t / 9) = Nc * (9 - t) / 9
                    # math.floor(x)返回小于参数x的最大整数
                    # threshold为c标签对应一个位置的值
                    idxs2 = idxs[pseudo_probs[idxs] > threshold]  # 伪标签对应位置大于此值的那些值
                    assert (selected[idxs2] == 0).all()
                    selected[idxs2] = 1  # 设置为1，表示命中
            return selected  # 返回已选择的标签
        selected = select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, params.T)  # pseudo_labels t为迭代的轮次1-10，T=10
        # pseudo_labels, pseudo_probs (2427,),(2427,) 对应伪标签值和数据值
        # 返回值selected (2427,)为已选择的标签
        selected = selected * (1 - rejected)  # rejected拒绝的标签，初始为全0的numpy数组，shape(2427,)
        # selected (2427,)为已选择的标签 原理：2427个中大部分为负样本，因此负样本损失值更小

        if t == 2:  # 第2轮
            def select_initial_rejected(pseudo_probs, n_r):
                # pseudo_probs为2427个伪标签对应的数据值，n_r=1200
                is_rejected = np.zeros((len(pseudo_probs),), dtype=np.int)  # is_rejected为2427个元素的全0numpy数组
                is_rejected[np.argsort(pseudo_probs)[:n_r]] = 1  # np.argsort(pseudo_probs)为从小到大排序后对应的元素标签
                # [:n_r]]取前1200个，设置为1
                return is_rejected
            rejected = select_initial_rejected(pseudo_probs, params.n_r)  # pseudo_probs为2427个伪标签对应的数据值，n_r=1200
            # 选择初始拒绝标签(交叉熵损失)

        if t >= 2:
            def update_rejected(selected, rejected, features_T):
                # selected当前已选择标签(2427,), rejected(2427,), features_T(proj_T)(2427, 128)支持集特征
                unlabeled = (selected == 0) * (rejected == 0)  # 已选择未标记的标签或未选择的标签或未拒绝的标签为未标记的标签
                new_is_rejected = rejected.copy()  # 新拒绝的标签
                for idx in np.where(unlabeled)[0]:  # 对于未标记的标签
                    dist_to_selected = get_dist(features_T[idx], features_T[selected == 1]).min()  # get_dist为标准化处理
                    # features_T[idx]未标记标签信息，features_T[selected == 1]已选择标签信息，未标记标签信息广播后减去已选择标签信息，然后求l2范数
                    dist_to_rejected = get_dist(features_T[idx], features_T[rejected == 1]).min()
                    # features_T[idx]未标记标签信息，features_T[rejected == 1]已拒绝标签信息
                    if dist_to_rejected < dist_to_selected:  # 如果拒绝的数据更小，则新拒绝的设置为1
                        new_is_rejected[idx] = 1
                return new_is_rejected
            rejected = update_rejected(selected, rejected, proj_T)  # 更新拒绝标签
            # selected当前已选择标签(2427,), rejected(2427,), proj_T(2427, 128)
        selected = selected * (1 - rejected)  # 更新已选择标签，拒绝的为0

        pseudo_labels[selected == 0] = -1  # 未选择标签或已选择被拒绝的标签设置为-1(最终全部选择了)
        pseudo_labels[rejected == 1] = -2  # 已选择被拒绝的标签设置为-2(最终选择的负样本全部拒绝了)

    # final pseudo labels 最终的伪标签
    pseudo_labels[pseudo_labels == -2] = params.num_src_classes  # num_src_classes=25 拒绝的伪标签设置为25
    assert (pseudo_labels != -1).all()

    def evaluate(predicted, labels, num_src_classes):
        # predicted(pseudo_labels)伪标签(2427,), labels(lbls_T)大于25标签全部设置为25的Art数据集的标签(2427,), num_src_classes=25
        acc_unk = (predicted[labels == num_src_classes] == labels[labels == num_src_classes]).mean()  # art数据集负样本的准确率(布尔数组的平均值为准确率)
        accs = [(predicted[labels == c] == labels[labels == c]).mean() for c in range(num_src_classes)]  # 0-24标签的准确率
        acc_common = np.array(accs).mean()  # 0-24标签准确率的平均值
        hos = 2 * acc_unk * acc_common / (acc_unk + acc_common)
        _os = np.array(accs + [acc_unk]).mean()  # accs为list，因此此处为添加一个元素，再求平均值
        return f'OS={_os * 100:.2f} OS*={acc_common * 100:.2f} unk={acc_unk * 100:.2f} HOS={hos * 100:.2f}'
    # evaluation
    return evaluate(pseudo_labels, lbls_T, params.num_src_classes)
    # pseudo_labels伪标签, lbls_T大于25标签全部设置为25的Art数据集的标签(2427,), num_src_classes=25


if __name__ == '__main__':
    params = Params(pca_dim=512, proj_dim=128, T=10, n_r=1200,
                    dataset='OfficeHome', source='clipart', target='art',
                    num_src_classes=25, num_total_classes=65)
    print(params.source, params.target, main(params))  # clipart art OS=46.86 OS*=45.53 unk=80.19 HOS=58.08
    # main(params)输出值为OS=46.86 OS*=45.53 unk=80.19 HOS=58.08，是不确定的
