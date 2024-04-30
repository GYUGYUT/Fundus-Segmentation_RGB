from dataloader_dir2 import *
from sklearn.decomposition import PCA
def apply_pca_to_images(batch_images, n_components=3):
    """
    각 이미지에 PCA를 적용하여 주요 성분을 추출하는 함수.

    Parameters:
    - batch_images: 입력 이미지 배치. Shape는 (batch_size, channels, width, height).
    - n_components: 선택할 주요 성분 수 (default는 3).

    Returns:
    - transformed_images: PCA가 적용된 이미지 배치. Shape는 (batch_size, n_components, width, height).
    """

    batch_size, channels, width, height = batch_images.shape
    transformed_images = np.zeros((batch_size, n_components, width, height))

    # 각 이미지에 대해 PCA 적용
    for i in range(batch_size):
        image = batch_images[i].reshape(channels, -1).T  # 이미지를 2D로 변환
        pca = PCA(n_components=n_components)
        pca.fit(image)

        # PCA 변환
        transformed_image = pca.transform(image)

        # PCA가 적용된 이미지를 원래 형태로 변환하여 저장
        transformed_images[i] = transformed_image.T.reshape(n_components, width, height)

    return torch.from_numpy(transformed_images).float(), pca.explained_variance_ratio_

train_data_path = r"/home/gyutae/atops2019/hyperspectral_image/train"
val_data_path = r"/home/gyutae/atops2019/hyperspectral_image/valid"
test_data_path = r"/home/gyutae/atops2019/hyperspectral_image/test"

Label_train_data_path = r"/home/gyutae/atops2019/train_1.csv"
Label_val_data_path = r"/home/gyutae/atops2019/valid.csv"
Label_test_data_path = r"/home/gyutae/atops2019/test.csv"

train_data,val_data,test_data = get_loder_main(train_data_path,val_data_path,test_data_path,
                                                   Label_train_data_path,Label_val_data_path,Label_test_data_path,
                                                   [512,512],3,True,8)

for batch_id, (X, y) in enumerate(tqdm(train_data)):
    X1, z1 = apply_pca_to_images(X,3)
    X2, z2= apply_pca_to_images(X,4)
    X3, z3 = apply_pca_to_images(X,5)
    X4, z4 = apply_pca_to_images(X,6)
    X5, z5 = apply_pca_to_images(X,7)
    X6, z6 = apply_pca_to_images(X,8)
    X7, z7 = apply_pca_to_images(X,9)
    X8, z8 = apply_pca_to_images(X,10)
    print(z1, np.sum(z1))
    print(z2, np.sum(z2))
    print(z3, np.sum(z3))
    print(z4, np.sum(z4))
    print(z5, np.sum(z5))
    print(z6, np.sum(z6))
    print(z7, np.sum(z7))
    print(z8, np.sum(z8))
    break
# [0.8512041  0.09366193 0.03426456] 0.9791305878747251
# [0.8512041  0.09366193 0.03426456 0.01354118] 0.9926717701557711
# [0.8512041  0.09366193 0.03426456 0.01354118 0.00373932] 0.9964110906621731
# [0.8512041  0.09366193 0.03426456 0.01354118 0.00373932 0.00184754] 0.9982586350305828
# [8.51204104e-01 9.36619270e-02 3.42645573e-02 1.35411823e-02 3.73932051e-03 1.84754437e-03 6.84015220e-04] 0.9989426502510169
# [8.51204104e-01 9.36619270e-02 3.42645573e-02 1.35411823e-02 3.73932051e-03 1.84754437e-03 6.84015220e-04 3.88178554e-04] 0.9993308288047511
# [8.51204104e-01 9.36619270e-02 3.42645573e-02 1.35411823e-02 3.73932051e-03 1.84754437e-03 6.84015220e-04 3.88178554e-04 2.09412020e-04] 0.9995402408245395
# [8.51204104e-01 9.36619270e-02 3.42645573e-02 1.35411823e-02 3.73932051e-03 1.84754437e-03 6.84015220e-04 3.88178554e-04 2.09412020e-04 1.21337523e-04] 0.9996615783475149