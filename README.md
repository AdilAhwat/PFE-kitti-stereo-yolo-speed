# AutoDrive Vision

Projet de fin d’études d’estimation de la profondeur et de la vitesse des objets routiers depuis un véhicule en mouvement. Le pipeline traite des séquences KITTI Raw et produit des images annotées et des mesures par objet. Le dashboard Streamlit permet de parcourir ces résultats et de les comparer aux références KITTI.

## Architecture actuelle

| Étape | Méthode utilisée | Sortie |
| --- | --- | --- |
| Stéréo | Cartes FoundationStereo précalculées, ou StereoSGBM avec filtrage WLS | Disparité en pixels, puis profondeur métrique avec la calibration KITTI |
| Détection et segmentation | YOLO11m-seg sur la caméra gauche | Boîtes, classes et masques d’instance |
| Suivi | StrongSORT via BoxMOT avec réidentification OSNet | Identifiants de suivi |
| Localisation | Profondeur robuste dans le masque, puis projection du centre de la boîte | Position 3D par objet. La mesure utilise une zone centrale de la boîte si le masque manque |
| Mouvement | OXTS et filtre de Kalman 3D à bruit de mesure adaptatif | Vitesse relative, vitesse compensée et statut |
| Consultation | Export CSV, images annotées et Streamlit | Lecture vidéo, courbes par piste et rapports d’évaluation |

Avec la **méthode A**, le pipeline compense la translation en ajoutant la vitesse OXTS du véhicule à la vitesse relative, dans les axes caméra. Avec la **méthode B** (`--world-kalman`), il transforme les positions selon la rotation et la translation de la caméra, puis estime la vitesse dans un repère local fixe. Le vecteur compensé est réexprimé dans les axes de la caméra courante pour l’export. Ce changement d’axes conserve la norme du vecteur.

## Installation sous Windows

Ouvrez PowerShell à la racine du projet cloné. Si l’environnement `.venv` existe déjà, réutilisez-le.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

L’environnement local contient Python 3.13.2, BoxMOT 23.0.0, Ultralytics 8.4.131 et Streamlit 1.62.0. Les versions minimales des dépendances figurent dans [requirements.txt](requirements.txt). La configuration par défaut se trouve dans [config/default.yaml](config/default.yaml). Elle sélectionne `yolo11m-seg.pt` et le modèle ReID `osnet_x0_25_msmt17.pt` sur CPU. StereoSGBM/WLS nécessite les fonctions WLS du module `cv2.ximgproc` fourni par OpenCV contrib.

## Préparer les séquences KITTI

Utilisez les archives KITTI Raw **synced+rectified**, les calibrations de la date et, pour la validation par objet, les annotations **tracklets**. Les deux séquences utilisées dans les comparaisons sont **0005** (154 paires) et **0014** (314 paires).

```text
C:\BIG DATA\PFE\KITTI\2011_09_26\
├── calib_cam_to_cam.txt
├── calib_imu_to_velo.txt
├── calib_velo_to_cam.txt
└── 2011_09_26_drive_0005_sync\
    ├── image_02\data\0000000000.png
    ├── image_03\data\0000000000.png
    ├── oxts\data\0000000000.txt
    ├── tracklet_labels.xml
    └── velodyne_points\data\0000000000.bin
```

Les deux séquences du 26 septembre partagent les fichiers de calibration du dossier parent. Sélectionnez les images, les mesures OXTS et les tracklets du même trajet. Le script conserve les noms d’images à dix chiffres, par exemple `0000000000`.

- Les fichiers `calib_imu_to_velo.txt` et `calib_velo_to_cam.txt` sont requis pour la méthode B.
- Les tracklets sont facultatifs pour l’estimation. Leur projection pour la validation nécessite `calib_velo_to_cam.txt`.
- Les scans `velodyne_points/data` servent à comparer les disparités au LiDAR projeté.
- Les cartes FoundationStereo sont calculées hors du pipeline Python local. Chaque fichier NumPy `.npy` doit porter le même nom que l’image gauche, avoir ses dimensions et contenir une disparité en pixels à la résolution d’origine.

## Lancer le pipeline

Dans PowerShell, choisissez `0005` ou `0014` dans `$drive`. Exécutez ces blocs dans la même session, depuis la racine du projet.

```powershell
$python = '.\.venv\Scripts\python.exe'
$drive = '0005'
$dateDir = 'C:\chemin\vers\KITTI\2011_09_26'
$sequence = Join-Path $dateDir "2011_09_26_drive_${drive}_sync"
$maps = "outputs\foundationstereo_kitti_${drive}\disparities"

$common = @(
    '--left-dir', "$sequence\image_02\data"
    '--right-dir', "$sequence\image_03\data"
    '--oxts-dir', "$sequence\oxts\data"
    '--calib', "$dateDir\calib_cam_to_cam.txt"
    '--calib-imu-to-velo', "$dateDir\calib_imu_to_velo.txt"
    '--calib-velo-to-cam', "$dateDir\calib_velo_to_cam.txt"
    '--tracklets', "$sequence\tracklet_labels.xml"
)
```

Adaptez `$dateDir` au dossier KITTI de votre machine. Retirez la ligne `--tracklets` du bloc si vous n’avez pas les annotations. Lancez un essai de 30 images avec StereoSGBM/WLS et la méthode B pour vérifier votre installation.

```powershell
& $python scripts\run_sequence.py @common --world-kalman --max-frames 30 --output-dir "outputs\test_sgbm_${drive}_30"
```

Pour traiter le trajet complet avec FoundationStereo, placez les cartes exportées depuis Colab dans le dossier indiqué par `$maps`. Utilisez deux dossiers de sortie pour conserver les résultats de chaque méthode.

```powershell
$foundationOut = "outputs\run_foundation_${drive}_methode_b"
$sgbmOut = "outputs\run_sgbm_${drive}_methode_b"

& $python scripts\run_sequence.py @common --world-kalman --foundation-disparity-dir $maps --max-frames 0 --output-dir $foundationOut
& $python scripts\run_sequence.py @common --world-kalman --max-frames 0 --output-dir $sgbmOut
```

Sans `--foundation-disparity-dir`, le script calcule la disparité avec StereoSGBM/WLS. Pour tester la méthode A, retirez `--world-kalman` et choisissez un autre dossier de sortie. `--max-frames 0` traite toutes les paires disponibles.

## Dashboard AutoDrive Vision

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Dans la barre latérale, sélectionnez un dossier contenant `measurements.csv` et ses images annotées. Pour afficher les cartes FoundationStereo, renseignez **Dossier des cartes de disparité** avec le sous-dossier `disparities` du trajet et **Calibration KITTI** avec `calib_cam_to_cam.txt`.

| Onglet | Fonction |
| --- | --- |
| Détection | Parcourir les images avec les boîtes, classes, IDs, profondeurs et vitesses disponibles |
| Profondeur | Visualiser la carte métrique ou calculer la disparité d’une paire rectifiée avec le module stéréo local |
| Séquence | Lire et télécharger la vidéo des images annotées. Le réglage d’images par seconde agit sur la lecture |
| Suivi | Sélectionner un ID et examiner les courbes de profondeur et de vitesse |
| Analyse | Consulter la validation sur les tracklets, télécharger le CSV et comparer les résultats stéréo sur le LiDAR projeté |

Les dossiers `outputs/` de l’expérimentation restent sur la machine où les résultats ont été produits. Après avoir cloné le dépôt, lancez le pipeline pour générer des images et un CSV, ou récupérez ces fichiers séparément pour une démo. Le volet **Comparaison des modèles** lit les rapports `outputs/stereo_evaluation/kitti_*.json`. Pour consulter des cartes StereoSGBM/WLS dans le tableau de bord, renseignez le dossier de leurs `.npy` s’il vous a été fourni. Le traitement de séquence écrit les images annotées et le CSV.

## Lire et évaluer les résultats

Chaque traitement écrit des PNG annotés et un fichier `measurements.csv` séparé par des points-virgules. La vidéo est produite par le dashboard à la demande. Les graphiques peuvent aussi être créés avec [plot_results.py](scripts/plot_results.py).

| Champ CSV | Signification |
| --- | --- |
| `distance_m` | Profondeur dans l’axe avant Z de la caméra, en mètres |
| `relative_speed_km_h` | Composante longitudinale signée de la vitesse relative |
| `world_speed_km_h` | Composante longitudinale signée de la vitesse compensée |
| `world_vx_mps`, `world_vy_mps`, `world_vz_mps` | Composantes compensées en m/s, exprimées dans les axes caméra |
| `speed_3d_km_h` et `speed_3d_raw_km_h` | Norme 3D affichée et norme brute avant la décision d’immobilité |
| `speed_status` | `waiting` en attente, `accepted` accepté, `rejected` écarté ou `static` estimé immobile |
| `analysis_eligible` et `analysis_reason` | Décision et motif du filtre de zone et de mouvement. Les lignes exclues restent dans le CSV |

Un champ vide indique une valeur indisponible. Le niveau `high` indique une bonne cohérence selon les critères internes du filtre. La précision se mesure séparément avec une référence.

```powershell
$csv = Join-Path $foundationOut 'measurements.csv'
& $python scripts\report_validation.py --csv $csv
& $python scripts\evaluate_tracking.py --csv $csv --tracker StrongSORT
& $python scripts\plot_results.py --csv $csv
```

[report_validation.py](scripts/report_validation.py) calcule la MAE et le RMSE de profondeur et de vitesse relative sur les observations retenues. [evaluate_tracking.py](scripts/evaluate_tracking.py) compte les changements d’ID, les interruptions et les fragments à partir des associations aux tracklets. Pour tracer une piste précise, ajoutez `--track-id` et son numéro à `plot_results.py`.

[evaluate_stereo.py](scripts/evaluate_stereo.py) compare deux dossiers de disparités précalculées au LiDAR projeté. Il attend des fichiers `.npy` appariés pour les deux méthodes, passés par `--sgbm` et `--foundation`. Indiquez le dossier du trajet par `--sequence` et le chemin du rapport JSON par `--output`. Le rapport indique la proportion de pixels corrects (erreur inférieure à 3 px **ou** à 5 %), la couverture et l’erreur moyenne sur les pixels valides.

### Résultats sur les séquences 0005 et 0014

| Séquence | Pixels corrects sur le LiDAR, SGBM/WLS à FoundationStereo | MAE profondeur | MAE vitesse relative |
| --- | --- | --- | --- |
| 0005 | 71,91 % → 93,09 % | 1,07 m → 0,62 m (266 observations) | 1,60 km/h → 1,11 km/h (167 observations) |
| 0014 | 72,55 % → 93,60 % | 2,02 m → 1,52 m (527 observations) | 3,46 km/h → 2,79 km/h (251 observations) |

Le taux de pixels corrects couvre toutes les images du trajet et les pixels où le LiDAR fournit une référence. Les deux MAE comparent les méthodes sur les mêmes observations par objet.

Pour la vitesse 3D brute sur la 0005, la MAE passe de **7,12 à 1,52 km/h** entre les méthodes A et B sur 497 observations. La référence reconstruite à partir des tracklets et la méthode B utilisent les mêmes poses OXTS.

## Limites du prototype

- Le calcul temporel utilise une cadence nominale de 10 Hz. Un décalage ou une image manquante peut fausser la vitesse.
- La profondeur sous masque décrit une surface visible. La référence KITTI correspond à un point de l’objet annoté.
- Les grandes distances, les occultations et les reprises d’identité peuvent produire des vitesses erronées.
- Les poids FoundationStereo ont été affinés sur des groupes de frames des trajets étudiés. Une validation de généralisation demanderait un trajet indépendant.
