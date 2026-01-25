"""
Dataset for BigEarthNet dataset. Files can be requested by contacting
the author.
Original Paper of Image Data:
TO BE PUBLISHED

https://bigearth.net/
"""
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union

import pandas as pd
from torch.utils.data import Dataset

from configilm.extra.BENv2_utils import ben_19_labels_to_multi_hot
from configilm.extra.BENv2_utils import BENv2LDMBReader
from configilm.extra.BENv2_utils import stack_and_interpolate
from configilm.extra.BENv2_utils import STANDARD_BANDS
from configilm.util import Messages
# from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from configilm.extra.BENv2_utils import NEW_LABELS

import numpy as np
import torch

from torchvision.transforms import v2

import mgrs
from math import cos, radians

def get_lat_lon_from_mgrs(tile_id: str):
    if tile_id.startswith("T") and len(tile_id) == 6:
        mgrs_code = tile_id[1:]
    else:
        mgrs_code = tile_id

    m = mgrs.MGRS()
    try:
        lat, lon = m.toLatLon(mgrs_code)
        return float(lat), float(lon)
    except Exception as e:
        print(f"Error in get_lat_lon_from_mgrs({tile_id}): {e}")
        return None, None


def approx_latlon_from_patch_id_with_mgrs(patch_id: str, patch_size_m: float = 1200.0):
    parts = patch_id.split("_")

    tile_code = None
    for p in parts:
        if len(p) == 6 and p.startswith("T") and p[1:3].isdigit():
            tile_code = p
            break

    if tile_code is None:
        return -1.0, -1.0

    lat0, lon0 = get_lat_lon_from_mgrs(tile_code)
    if lat0 is None or lon0 is None:
        return -1.0, -1.0

    try:
        col = int(parts[-2])
        row = int(parts[-1])
    except ValueError:
        return lat0, lon0

    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = 111_000.0 * cos(radians(lat0))

    dlat_per_patch = patch_size_m / meters_per_deg_lat
    dlon_per_patch = patch_size_m / meters_per_deg_lon if meters_per_deg_lon != 0 else 0.0

    lat_sw = lat0 + row * dlat_per_patch
    lon_sw = lon0 + col * dlon_per_patch

    lat_center = lat_sw + 0.5 * dlat_per_patch
    lon_center = lon_sw + 0.5 * dlon_per_patch

    return float(lat_center), float(lon_center)

def get_season_from_patch_id(patch_id: str) -> int:
    """
    Parses the patch_id to extract the date and classify it into a season.
    Returns an integer code for the season.
    0: Spring (Mar, Apr, May)
    1: Summer (Jun, Jul, Aug)
    2: Autumn (Sep, Oct, Nov)
    3. Winter (Dec, Jan, Feb)
    """
    try:
        # Sentinel-2: S2A_MSIL2A_20170613T101031_...
        # Sentinel-1: S1A_IW_GRDH_1SDV_20170613T171705_...
        # 날짜 부분은 항상 비슷한 위치에 있습니다.
        date_str = ""
        parts = patch_id.split('_')
        for part in parts:
            if part.startswith('20') and 'T' in part: # '20170613T101031' 같은 형식 찾기
                date_str = part.split('T')[0]
                break
        
        if not date_str:
            return -1 # 날짜를 찾지 못한 경우 (오류)

        month = int(date_str[4:6])
        if 3 <= month <= 5:
            return 0  # Spring
        elif 6 <= month <= 8:
            return 1  # Summer
        elif 9 <= month <= 11:
            return 2  # Autumn
        else: # 12, 1, 2
            return 3  # Winter
    except (ValueError, IndexError):
        # 예외 발생 시 -1 반환
        return -1


class BENv2DataSet(Dataset):
    """
    Dataset for BigEarthNet dataset. LMDB-Files can be requested by contacting
    the author or by downloading the dataset from the official website and encoding
    it using the BigEarthNet Encoder.

    The dataset can be loaded with different channel configurations. The channel configuration
    is defined by the first element of the img_size tuple (c, h, w).
    The available configurations are:

        - 2 -> Sentinel-1 (VV, VH)
        - 3 -> RGB
        - 4 -> 10m Sentinel-2 (B, R, G, Ir)
        - 10 -> 10m + 20m Sentinel-2 (in original order)
        - 12 -> Sentinel-1 + 10m/20m Sentinel-2 (in original order)
        - 14 -> Sentinel-1 + 10m/20m/60m Sentinel-2 (in original order)

    Original order means that the bands are ordered as they are defined by ESA:
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    In detail, this means:
        - 2: VV, VH
        - 3: B04, B03, B02
        - 4: B02, B03, B04, B08
        - 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        - 12: VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        - 14: VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    """

    avail_chan_configs = {
        2: "Sentinel-1",
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2 (in original order)",
        12: "Sentinel-1 + 10m + 20m Sentinel-2 (in original order)",
        14: "Sentinel-1 + 10m + 20m + 60m Sentinel-2 (in original order)",
    }

    channel_configurations = {
        2: STANDARD_BANDS["S1"],
        3: STANDARD_BANDS["RGB"],  # RGB order
        4: STANDARD_BANDS["10m"],  # BRGIr order
        10: STANDARD_BANDS["10m_20m"],  # Original order
        12: STANDARD_BANDS["S1_10m_20m"],  # Original order
        14: STANDARD_BANDS["ALL"],  # Original order
    }

    @classmethod
    def get_available_channel_configurations(cls):
        """
        Prints all available preconfigured channel combinations.
        """
        Messages.hint("Available channel configurations are:")
        for c, m in cls.avail_chan_configs.items():
            Messages.hint(f"    {c:>3} -> {m}")


    def __init__(
        self,
        data_dirs: Mapping[str, Union[str, Path]],
        split: Optional[str] = None,
        transform = None,
        max_len: Optional[int] = None,
        img_size: tuple = (3, 120, 120),
        return_extras: bool = False,
        patch_prefilter: Optional[Callable[[str], bool]] = None,
        include_cloudy: bool = False,
        include_snowy: bool = False,
        merge_patch: bool = False,
        get_labels_name: bool = False,
        return_season: bool = False,
        return_diffsat_metadata: bool = False,
    ):
        """
        Dataset for BigEarthNet v2 dataset. Files can be requested by contacting
        the author or visiting the official website.

        Original Paper of Image Data:
        TO BE PUBLISHED

        :param data_dirs: A mapping from file key to file path. The file key is
            used to identify the function of the file. The required keys are:
            "images_lmdb", "metadata_parquet", "metadata_snow_cloud_parquet".

        :param split: The name of the split to use. Can be either "train", "val" or
            "test". If None is provided, all splits are used.

            :default: None

        :param transform: A callable that is used to transform the images after
            loading them. If None is provided, no transformation is applied.

            :default: None

        :param max_len: The maximum number of images to use. If None or -1 is
            provided, all images are used.

            :default: None

        :param img_size: The size of the images. Note that this includes the number of
            channels. For example, if the images are RGB images, the size should be
            (3, h, w).

            :default: (3, 120, 120)

        :param return_extras: If True, the dataset will return the patch name
            as a third return value.

            :default: False

        :param patch_prefilter: A callable that is used to filter the patches
            before they are loaded. If None is provided, no filtering is
            applied. The callable must take a patch name as input and return
            True if the patch should be included and False if it should be
            excluded from the dataset.

            :default: None
        """
        super().__init__()
        self.return_extras = return_extras
        self.lmdb_dir = data_dirs["images_lmdb"]
        self.transform = transform
        self.image_size = img_size
        self.merge_patch = merge_patch
        self.get_labels_name = get_labels_name
        self.return_diffsat_metadata = return_diffsat_metadata
        assert len(img_size) == 3, "Image size must be a tuple of length 3"
        c, h, w = img_size
        assert h == w, "Image size must be square"
        if c not in self.avail_chan_configs.keys():
            BENv2DataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        Messages.info(f"Loading BEN data for {split}...")
        # read metadata
        metadata = pd.read_parquet(data_dirs["metadata_parquet"])
        # print(metadata.columns.tolist())


        self.patch_metadata = {}
        for _, row in metadata.iterrows():
            pid = row["patch_id"]
            
            # 1. 위도/경도 (기존 코드 유지)
            # lat = row.get("center_lat", None)
            # lon = row.get("center_lon", None)
            lat, lon = approx_latlon_from_patch_id_with_mgrs(pid)

            # 2. 날짜 정보 (patch_id에서 직접 파싱)
            year = month = day = None
            
            try:
                # patch_id 예시: S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_48_74
                # 언더바(_)로 자르면 3번째 요소([2])가 날짜입니다.
                parts = pid.split("_")
                
                if len(parts) >= 3:
                    date_part = parts[2]  # "20170613T101031" 가져오기
                    
                    # 'T' 기준으로 앞부분만 자름 -> "20170613"
                    if "T" in date_part:
                        date_str = date_part.split("T")[0]
                        
                        # 길이가 8자리(YYYYMMDD)인지 확인 후 파싱
                        if len(date_str) == 8:
                            year = int(date_str[0:4])   # 2017
                            month = int(date_str[4:6])  # 06
                            day = int(date_str[6:8])    # 13

            except (ValueError, IndexError):
                # 파싱 실패 시 None 유지
                pass

            self.patch_metadata[pid] = {
                "lat": lat,
                "lon": lon,
                "year": year,
                "month": month,
                "day": day,
                # "country": row.get("country", None),
            }
        if include_cloudy or include_snowy:
            metadata_snow_cloud = pd.read_parquet(data_dirs["metadata_snow_cloud_parquet"])
            metadata = pd.concat([metadata, metadata_snow_cloud])
        if not include_cloudy:
            # remove all rows with contains_cloud_or_shadow
            metadata = metadata[~metadata["contains_cloud_or_shadow"]]
        if not include_snowy:
            # remove all rows with contains_seasonal_snow
            metadata = metadata[~metadata["contains_seasonal_snow"]]
        if split is not None:
            metadata = metadata[metadata["split"] == split]
        self.patches = metadata["patch_id"].tolist()
        self.labels = metadata["labels"].tolist()
        self.return_season = return_season

        Messages.info(f"    {len(self.patches)} patches indexed")

        # if a prefilter is provided, filter patches based on function
        if patch_prefilter:
            self.patches = list(filter(patch_prefilter, self.patches))
        Messages.info(f"    {len(self.patches)} pre-filtered patches indexed")

        # sort list for reproducibility
        self.patches.sort()
        if max_len is not None and max_len < len(self.patches) and max_len != -1:
            self.patches = self.patches[:max_len]
        Messages.info(f"    {len(self.patches)} filtered patches indexed")

        #################### modify for sar2opt ####################
        if self.merge_patch:    
            self.patch_info = {}  # (tile, row, col) -> patch_id
            all_patches_set = set(self.patches) # 빠른 포함 여부 확인용

            for patch_id in self.patches:
                parts = patch_id.split('_')
                col, row = int(parts[-2]), int(parts[-1])
                tile_id = "_".join(parts[:-2])
                self.patch_info[(tile_id, col, row)] = patch_id
            
            composite_patches = []
            for patch_id in self.patches:
                parts = patch_id.split('_')
                col, row = int(parts[-2]), int(parts[-1]) # 순서 변경! col, row
                tile_and_date_id = "_".join(parts[:-2])
                
                has_right = (tile_and_date_id, col + 1, row) in self.patch_info
                has_down = (tile_and_date_id, col, row + 1) in self.patch_info
                has_down_right = (tile_and_date_id, col + 1, row + 1) in self.patch_info
                
                if has_right and has_down and has_down_right:
                    composite_patches.append(patch_id)

            self.patches = composite_patches
            self.patches.sort() # 재현성을 위해 다시 정렬
            
            if max_len is not None and max_len < len(self.patches) and max_len != -1:
                self.patches = self.patches[:max_len]
            
            Messages.info(f"    {len(self.patches)} composite (2x2) patches indexed")

        
        self.pos_weight = self._compute_class_weights()
        Messages.info("Class weights calculated.")

        self.channel_order = self.channel_configurations[c]
        # self.BENv2Loader = BENv2LDMBReader(
        #     image_lmdb_file=self.lmdb_dir,
        #     metadata_file=data_dirs["metadata_parquet"],
        #     metadata_snow_cloud_file=data_dirs["metadata_snow_cloud_parquet"],
        #     bands=self.channel_order,
        #     process_bands_fn=partial(stack_and_interpolate, img_size=h, upsample_mode="nearest"),
        #     process_labels_fn=ben_19_labels_to_multi_hot,
        # )
        self.loader_kwargs = {
            "image_lmdb_file": self.lmdb_dir,
            "metadata_file": data_dirs["metadata_parquet"],
            "metadata_snow_cloud_file": data_dirs["metadata_snow_cloud_parquet"],
            "bands": self.channel_order,
            "process_bands_fn": partial(stack_and_interpolate, img_size=h, upsample_mode="nearest"),
            "process_labels_fn": ben_19_labels_to_multi_hot,
        }
        self.BENv2Loader = None


    def get_patchname_from_index(self, idx: int) -> Optional[str]:
        """
        Gives the patch name of the image at the specified index. May return invalid
        names (names that are not actually loadable because they are not part of the
        lmdb file) if the name is included in the metadata file(s).

        :param idx: index of an image
        :return: patch name of the image or None, if the index is invalid
        """
        if idx > len(self):
            return None
        return self.patches[idx]


    def _build_diffsat_metadata(self, patch_id: str) -> dict:
        md = self.patch_metadata.get(patch_id, {})

        lon = md.get("lon", None)
        lat = md.get("lat", None)
        year = md.get("year", None)
        month = md.get("month", None)
        day = md.get("day", None)

        # print(md.get("country", None))

        gsd = 10.0
        cloud_cover = 0.0

        def _none_to_minus1(x):
            return -1 if x is None else x

        lon = _none_to_minus1(lon)
        lat = _none_to_minus1(lat)
        year = _none_to_minus1(year)
        month = _none_to_minus1(month)
        day = _none_to_minus1(day)

        return {
            "lon": lon,
            "lat": lat,
            "gsd": gsd,
            "cloud_cover": cloud_cover,
            "year": year,
            "month": month,
            "day": day,
        }

    def get_index_from_patchname(self, patchname: str) -> Optional[int]:
        """
        Gives the index of the image of a specific name. Does not distinguish between
        invalid names (not in original BigEarthNet) and names not in loaded list.

        :param patchname: name of an image
        :return: index of the image or None, if the name is not loaded
        """
        if patchname not in set(self.patches):
            return None
        return self.patches.index(patchname)

    def _compute_class_weights(self):
        """
        Calculates positive weights for BCEWithLogitsLoss.
        Formula: weight_c = (N - count_c) / count_c
        """
        # NEW_LABELS는 알파벳 순서로 정렬된 19개 클래스 리스트여야 합니다.
        label_map = {lbl: idx for idx, lbl in enumerate(NEW_LABELS)}
        num_classes = len(NEW_LABELS)
        
        # 카운트 초기화
        counts = np.zeros(num_classes)
        
        # 전체 라벨 순회 (self.labels는 split 필터링이 적용된 상태)
        # 주의: max_len 등으로 잘리기 전의 해당 split 전체 분포를 사용하는 것이 일반적입니다.
        for label_list in self.labels:
            for lbl in label_list:
                if lbl in label_map:
                    counts[label_map[lbl]] += 1
        
        # 전체 샘플 수
        N = len(self.labels)
        
        # 0으로 나누기 방지
        counts = np.clip(counts, a_min=1, a_max=None)
        
        # 가중치 계산 (Negative / Positive)
        pos_weight = (N - counts) / counts
        
        # Clipping (가중치가 너무 크면 학습 불안정, 1~100 사이로 제한)
        pos_weight = np.clip(pos_weight, a_min=1.0, a_max=100.0)
        
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_class_weights(self):
        """External method to retrieve calculated weights"""
        return self.pos_weight


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):

        if self.BENv2Loader is None:
            self.BENv2Loader = BENv2LDMBReader(**self.loader_kwargs)

        tl_key = self.patches[idx]
        
        seed = torch.seed() 

        if self.merge_patch:        
            parts = tl_key.split('_')
            col, row = int(parts[-2]), int(parts[-1])
            tile_and_date_id = "_".join(parts[:-2])
            
            tr_key = self.patch_info[(tile_and_date_id, col + 1, row)]     # 우상단
            bl_key = self.patch_info[(tile_and_date_id, col, row + 1)]     # 좌하단
            br_key = self.patch_info[(tile_and_date_id, col + 1, row + 1)] # 우하단

            
            img_tl, labels_tl = self.BENv2Loader[tl_key]
            img_tr, labels_tr = self.BENv2Loader[tr_key]
            img_bl, labels_bl = self.BENv2Loader[bl_key]
            img_br, labels_br = self.BENv2Loader[br_key]

            top_row = np.concatenate((img_tl, img_tr), axis=2)  # width-wise
            bottom_row = np.concatenate((img_bl, img_br), axis=2) # width-wise
            img = np.concatenate((top_row, bottom_row), axis=1) # height-wise
            # 최종 img shape: (channels, 240, 240)
            
            labels = np.logical_or.reduce([labels_tl, labels_tr, labels_bl, labels_br]).astype(np.float32)
        else:
            img, labels = self.BENv2Loader[tl_key]
            img = img.detach().numpy()
            labels = labels.detach().numpy()

        # print(tl_key)
        season_code = get_season_from_patch_id(tl_key)


        sar_channels = torch.Tensor(img[[0, 1], :, :])
        opt_channels = torch.Tensor(img[[4, 3, 2], :, :])

        if self.transform:
            sar = self.transform["sar"]["preprocess"](sar_channels)
            opt = self.transform["opt"]["preprocess"](opt_channels)
            
            combined = torch.cat([sar, opt], dim=0)

            shared_augment = self.transform["opt"]["augment"]
            augmented_combined = shared_augment(combined)
            
            # print('augmented_combine.shape', augmented_combined.shape)
            
            if augmented_combined.shape[0] == 5:
                sar = augmented_combined[:2, :, :]
                opt = augmented_combined[2:, :, :]
            else:
                sar = augmented_combined[:3, :, :]
                opt = augmented_combined[3:, :, :]

            # print('sar.shape after augment:', sar.shape)
            # print('opt.shape after augment:', opt.shape)
            
            
            sar = self.transform["sar"]["normalize"](sar)
            opt = self.transform["opt"]["normalize"](opt)

            img = {"sar": sar, "opt": opt}
        else:
            img = torch.from_numpy(img)


        if self.return_extras:
            return img, labels, tl_key

        if getattr(self, "return_diffsat_metadata", False):
            diffsat_md = self._build_diffsat_metadata(tl_key)
            return img, labels, diffsat_md
        
        if self.get_labels_name:
            return img, labels, ','.join(self.labels[idx].tolist())
        return img, labels