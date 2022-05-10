import os
import os.path
import sys
import h5py
from matplotlib import pyplot as plt

#入力したファイルのパス引数に渡す
input_file_name = sys.argv[1]
#h5ファイルの読み取り
f = h5py.File(input_file_name, 'r')

#予測した画像のデータセットの取得
dset = f['predictions']
#テスト、トレーニングで利用した画像のデータセットの取得
#dset = f['raw']
print(dset)

channel_num, x_num, y_num, z_num = map(int, dset.shape)
print(channel_num,x_num, y_num, z_num)

"""""
#テスト、トレーニング画像用
x_num, y_num, z_num = map(int,dset.shape)
print(x_num, y_num, z_num)
part = dset[0:320,0:960,500]

# plt.imshow(part,cmap='gray')
# plt.show()
"""""

#ファイル名の取得(ファイル名から.h5除いたもの)
name = os.path.basename(input_file_name)[0:-3]
#ファイル名でのディレクトリ作成
os.mkdir(name)


slice_step = 1
save_path = f'./{name}/'

# xを動かしてyz面のスライスの画像を保存
for i in range(0, x_num, slice_step):
    image = dset[0][i][0:y_num][0:z_num]
    filename = f'slice{str(i).zfill(5)}.tiff'
    #plt.imsave(save_path+filename, image)
    plt.imshow(image, cmap='gray')
    plt.savefig(save_path+filename)
    print(f'save at {os.path.normpath(save_path+filename)}')


#plt.imshow(part)
# plt.show()

