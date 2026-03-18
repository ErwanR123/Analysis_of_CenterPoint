# Analyse de CenterPoint — Center-based 3D Object Detection and Tracking

Analyse approfondie du papier de recherche [CenterPoint (CVPR 2021)](https://arxiv.org/abs/2006.11275) de Yin et al., avec dissection du pipeline, inférence sur le dataset nuScenes et discussion critique.

*Auteur : Erwan Ouabdesselam*

---

## Contenu du notebook

Le notebook `01_Analysis_and_Experiments.ipynb` est organisé en trois parties :

**I. Analyse du pipeline**
1. Mesures LiDAR et représentation BEV
2. Voxelisation et encodage MeanVFE
3. Backbone 3D sparse (VoxelNet) et neck 2D
4. Raffinement two-stage
5. Velocity head et tracking glouton

**II. Expériences**
1. Métriques d'évaluation nuScenes (mAP, NDS, TP errors)
2. Résultats du papier original
3. Nos résultats d'inférence sur nuScenes mini val
4. Expérimentations supplémentaires

**III. Discussion critique**

---

## Prérequis

### 1. Dataset nuScenes

Ce projet utilise la version **mini** du dataset nuScenes (`v1.0-mini`).

1. Créer un compte sur [nuscenes.org](https://www.nuscenes.org/) et télécharger `v1.0-mini`.
2. Placer les données dans `centerpoint/data/nuScenes/` avec la structure suivante :

```
centerpoint/data/nuScenes/
├── samples/          ← key frames
├── sweeps/           ← frames sans annotation
├── maps/
└── v1.0-mini/        ← métadonnées et annotations
```

3. Générer le fichier d'infos de validation (depuis `centerpoint/`) :

```bash
cd centerpoint
python tools/create_data.py nuscenes_data_prep \
    --root_path=data/nuScenes \
    --version="v1.0-mini" \
    --nsweeps=10
```

Cela crée `centerpoint/data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl`.

### 2. Poids du modèle pré-entraîné

Télécharger le checkpoint `centerpoint_voxel_1440_flip` depuis le [Model Zoo officiel](https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/README.md) et le placer ici :

```
work_dirs/nusc_centerpoint/epoch_20.pth
```

---

## Installation

### Environnement conda

```bash
conda create --name centerpoint python=3.6
conda activate centerpoint
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```

### Dépendances Python

```bash
cd centerpoint
pip install -r requirements.txt
```

### Extensions CUDA (obligatoires)

```bash
# Rotated NMS
cd centerpoint/det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

### spconv

```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
```

### nuscenes-devkit

```bash
pip install nuscenes-devkit
```

Pour l'installation complète (APEX, DCN, etc.), voir [centerpoint/docs/INSTALL.md](centerpoint/docs/INSTALL.md).

---

## Lancer le notebook

```bash
conda activate centerpoint
jupyter notebook 01_Analysis_and_Experiments.ipynb
```

Exécuter les cellules dans l'ordre. La cellule d'initialisation (cellule 2) charge le modèle, le dataset et définit tous les chemins.

---

## Structure du repo

```
Analysis_of_CenterPoint/
├── 01_Analysis_and_Experiments.ipynb   ← notebook principal
├── centerpoint/                         ← code source CenterPoint (fork officiel)
│   ├── configs/nusc/voxelnet/           ← config utilisée
│   ├── data/nuScenes/                   ← dataset (non versionné)
│   ├── det3d/                           ← framework de détection
│   └── tools/                           ← scripts d'entraînement/évaluation
├── figures/                             ← figures générées par le notebook
│   ├── bev_view.png
│   ├── centerhead_heatmaps.png
│   ├── full_pipeline.png
│   └── ...
└── work_dirs/
    ├── nusc_centerpoint/
    │   └── epoch_20.pth                 ← poids pré-entraînés (non versionnés)
    └── test/
        ├── metrics_summary.json         ← résultats d'inférence
        ├── examples/                    ← visualisations BEV de détections
        └── plots/                       ← courbes PR et TP errors par classe
```

---

## Résultats obtenus

| Modèle | Split | mAP (%) | NDS (%) |
|--------|-------|---------|---------|
| CenterPoint-VoxelNet (papier, Table 6) | nuScenes full val | 56.4 | 64.8 |
| CenterPoint-VoxelNet (notre inférence) | nuScenes mini val | **55.4** | — |

---

## Référence

```bibtex
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={CVPR},
  year={2021},
}
```
