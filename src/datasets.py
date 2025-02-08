from itertools import product
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import convert_to_8bit

class CZII2DDataset(Dataset):
    def __init__(self,
                 df,
                 data_dicts,
                 indices_ch,
                 direction='xy',
                 mode='valid',
                 transform=None
                 ):

        self.df = df
        self.data_dicts = data_dicts
        self.indices_ch = indices_ch
        self.direction = direction
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src2idx = {
            'denoised': 0,
            'ext_wbp': 1
            }
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        source = row['source']
        source_idx = src2idx[source]
        df_sample = self.df[self.df['sample_name'] == sample_name]
        n_slice = len(df_sample)
        slice_idx = row['slice_index']
        slice_indices = [np.clip(i+slice_idx, 0, n_slice-1) for i in self.indices_ch]
        img = self.data_dicts[sample_name]['image']
        lbl = self.data_dicts[sample_name]['label']
        if self.direction=='xy':
            img = img[slice_indices].transpose(1,2,0)
            lbl = lbl[slice_idx]
        else:
            if np.random.random()>0.5:
                img = img[:,:,slice_indices]
                lbl = lbl[:,:,slice_idx]
            else:
                img = img[:,slice_indices,:].transpose(0,2,1)
                lbl = lbl[:,slice_idx,:]
        img = convert_to_8bit(img, 1, 99)
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl)
            img = aug['image']
            lbl = aug['mask']
        img = img.astype(np.float32)/255.
        img = torch.tensor(img.transpose(2,0,1))
        lbl = torch.tensor(lbl, dtype=torch.long)
        source_idx = torch.tensor(source_idx, dtype=torch.float32)
        return {'image': img,
                'label': lbl,
                'source': source_idx
                }


class CZII2Dto3DDataset(Dataset):
    def __init__(self,
                 df,
                 data_dicts,
                 indices_ch,
                 direction='xy',
                 mode='valid',
                 transform=None
                 ):

        self.df = df
        self.data_dicts = data_dicts
        self.indices_ch = indices_ch
        self.direction = direction
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src2idx = {
            'denoised': 0,
            'ext_wbp': 1
            }
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        source = row['source']
        source_idx = src2idx[source]
        df_sample = self.df[self.df['sample_name'] == sample_name]
        n_slice = len(df_sample)
        slice_idx = row['slice_index']
        slice_indices = [np.clip(i+slice_idx, 0, n_slice-1) for i in self.indices_ch]
        img = self.data_dicts[sample_name]['image']
        lbl = self.data_dicts[sample_name]['label']
        if self.direction=='xy':
            img = img[slice_indices].transpose(1,2,0)
            lbl = lbl[slice_indices].transpose(1,2,0)
        else:
            if self.mode=='train':
                if np.random.random()>0.5:
                    img = img[:,:,slice_indices]
                    lbl = lbl[:,:,slice_indices]
                else:
                    img = img[:,slice_indices,:].transpose(0,2,1)
                    lbl = lbl[:,slice_indices,:].transpose(0,2,1)
            else:
                if self.direction=='yz':
                    img = img[:,:,slice_indices]
                    lbl = lbl[:,:,slice_indices]
                elif self.direction=='zx':
                    img = img[:,slice_indices,:].transpose(0,2,1)
                    lbl = lbl[:,slice_indices,:].transpose(0,2,1)
        img = convert_to_8bit(img, 1, 99)
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl)
            img = aug['image']
            lbl = aug['mask']
        img = img.astype(np.float32)/255.
        img = torch.tensor(img.transpose(2,0,1))
        lbl = torch.tensor(lbl.transpose(2,0,1), dtype=torch.long)
        source_idx = torch.tensor(source_idx, dtype=torch.float32)
        return {'image': img,
                'label': lbl,
                'source': source_idx
                }


class CZII2Dto3DXYZDataset(Dataset):
    def __init__(self,
                 df,
                 data_dicts,
                 indices_ch,
                 direction='xy',
                 mode='valid',
                 transform=None
                 ):

        self.df = df
        self.data_dicts = data_dicts
        self.indices_ch = indices_ch
        self.direction = direction
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src2idx = {
            'denoised': 0,
            'ext_wbp': 1
            }
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        source = row['source']
        source_idx = src2idx[source]
        df_sample = self.df[self.df['sample_name'] == sample_name]
        n_slice = len(df_sample)
        slice_idx = row['slice_index']
        slice_indices = [np.clip(i+slice_idx, 0, n_slice-1) for i in self.indices_ch]
        img = self.data_dicts[sample_name]['image']
        lbl = self.data_dicts[sample_name]['label']
        if self.mode=='train':
            if np.random.random()<0.33:
                # yz
                img = img[:,:,slice_indices]
                lbl = lbl[:,:,slice_indices]
            elif np.random.random()<0.5:
                # zx
                img = img[:,slice_indices,:].transpose(0,2,1)
                lbl = lbl[:,slice_indices,:].transpose(0,2,1)
            else:
                # xy
                img = img[slice_indices].transpose(1,2,0)
                lbl = lbl[slice_indices].transpose(1,2,0)
        else:
            if self.direction=='yz':
                img = img[:,:,slice_indices]
                lbl = lbl[:,:,slice_indices]
            elif self.direction=='zx':
                img = img[:,slice_indices,:].transpose(0,2,1)
                lbl = lbl[:,slice_indices,:].transpose(0,2,1)
            else:
                img = img[slice_indices].transpose(1,2,0)
                lbl = lbl[slice_indices].transpose(1,2,0)
        img = convert_to_8bit(img, 1, 99)
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl)
            img = aug['image']
            lbl = aug['mask']
        img = img.astype(np.float32)/255.
        img = torch.tensor(img.transpose(2,0,1))
        lbl = torch.tensor(lbl.transpose(2,0,1), dtype=torch.long)
        source_idx = torch.tensor(source_idx, dtype=torch.float32)
        return {'image': img,
                'label': lbl,
                'source': source_idx
                }
    

class CZII3DDataset(Dataset):
    def __init__(self,
                 df,
                 data_dicts,
                 indices_ch,
                 crop_fn,
                 mode='valid',
                 transform=None
                 ):

        self.df = df
        self.data_dicts = data_dicts
        self.indices_ch = indices_ch
        self.crop_fn = crop_fn
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row['sample_name']
        df_sample = self.df[self.df['sample_name'] == sample_name]
        n_slice = len(df_sample)
        slice_idx = row['slice_index']
        slice_indices = [np.clip(i+slice_idx, 0, n_slice-1) for i in self.indices_ch]
        img = self.data_dicts[sample_name]['image'][slice_indices].transpose(1,2,0)
        lbl = self.data_dicts[sample_name]['label'][slice_indices].transpose(1,2,0)
        img = convert_to_8bit(img, 1, 99)
        data = self.crop_fn(image=img, mask=lbl)
        if self.transform is not None:
            img = data['image'][np.newaxis]
            lbl = data['mask']
            aug = self.transform({'image': img, 'mask': lbl})
            img = aug['image']
            lbl = aug['mask']
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)/255.
            img = torch.tensor(img)
        else:
            img = img.to(torch.float32)/255.
        if isinstance(lbl, np.ndarray):
            lbl = torch.tensor(lbl, dtype=torch.long)
        else:
            lbl = lbl.to(dtype=torch.long)
        return {'image': img,
                'label': lbl
                }
    

class CZIIScannerDataset(Dataset):
    def __init__(self,
                 df,
                 data_dicts,
                 crop_size,
                 overlap_size,
                 image_shape,
                 direction='xy',
                 mode='valid',
                 transform=None
                 ):

        self.df = df
        self.data_dicts = data_dicts
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        self.image_shape=image_shape
        self.direction = direction
        self.mode = mode
        self.transform = transform
        # 各軸のスタート位置を計算
        self.z_starts = self._compute_start_positions(
            image_dim=image_shape[0],
            crop_dim=crop_size[0],
            overlap_dim=overlap_size[0]
        )
        self.y_starts = self._compute_start_positions(
            image_dim=image_shape[1],
            crop_dim=crop_size[1],
            overlap_dim=overlap_size[1]
        )
        self.x_starts = self._compute_start_positions(
            image_dim=image_shape[2],
            crop_dim=crop_size[2],
            overlap_dim=overlap_size[2]
        )

        # すべての (z, y, x) の組み合わせを生成
        self.positions = list(product(self.z_starts, self.y_starts, self.x_starts))
        self.current = 0

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        z_s, y_s, x_s = self.positions[idx]
        sample_name = self.df.iloc[idx]['sample_name']
        data = self.data_dicts[sample_name]
        img = data['image'][z_s:z_s+self.crop_size[0],y_s:y_s+self.crop_size[1],x_s:x_s+self.crop_size[2]]#.transpose(1,2,0)
        lbl = data['label'][z_s:z_s+self.crop_size[0],y_s:y_s+self.crop_size[1],x_s:x_s+self.crop_size[2]]#.transpose(1,2,0)
        img = convert_to_8bit(img, 1, 99)
        if self.direction=='xy':
            img = img.transpose(1,2,0)
            lbl = lbl.transpose(1,2,0)
        elif self.direction=='yz':
            pass
        elif self.direction=='zx':
            img = img.transpose(0,2,1)
            lbl = lbl.transpose(0,2,1)

        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl)
            img = aug['image']
            lbl = aug['mask']
        img = img.astype(np.float32)/255.
        img = torch.tensor(img.transpose(2,0,1))
        lbl = torch.tensor(lbl.transpose(2,0,1), dtype=torch.long)
        return {'image': img,
                'label': lbl,
                'z': z_s,
                'y': y_s,
                'x': x_s
                }
    def _compute_start_positions(self, image_dim, crop_dim, overlap_dim):
        """
        各軸のスタート位置を計算するヘルパー関数。

        Parameters:
        - image_dim: 画像のその軸のサイズ
        - crop_dim: クロップサイズのその軸のサイズ
        - overlap_dim: オーバーラップサイズのその軸のサイズ

        Returns:
        - starts: スタート位置のリスト
        """
        step = crop_dim - overlap_dim
        if step <= 0:
            raise ValueError("クロップサイズはオーバーラップサイズより大きくなければなりません。")

        starts = list(range(0, image_dim - crop_dim + 1, step))
        if not starts:
            # 画像がクロップサイズより小さい場合
            starts = [0]
        elif starts[-1] + crop_dim < image_dim:
            # 最後のクロップが画像端に達するように調整
            starts.append(image_dim - crop_dim)
        return starts