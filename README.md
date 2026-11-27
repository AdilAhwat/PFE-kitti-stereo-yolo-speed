# kitti_stereo_yolo_speed

Projet PFE d'estimation de profondeur et de vitesse des objets routiers, à partir d'images stéréo KITTI.

## Chaîne de traitement

```text
KITTI Raw : image_02 (gauche) + image_03 (droite)
                    ↓
StereoSGBM + filtre WLS : carte de disparité
                    ↓
Profondeur : Z = (f × B) / disparité
                    ↓
YOLO11m : détection des objets
                    ↓
BoT-SORT + ReID : ID persistant par objet et compensation caméra
                    ↓
Profondeur robuste dans la bounding box
                    ↓
OXTS : mouvement de la caméra
                    ↓
Filtre de Kalman 3D
                    ↓
Vecteur vitesse 3D (Vx, Vy, Vz)
                    ↓
Images annotées, CSV et graphiques
```

Le vecteur est exprimé dans le repère de la caméra rectifiée :

```text
X : droite / gauche
Y : bas / haut
Z : avant / arrière
V = (Vx, Vy, Vz) en m/s
```

La norme `|V|` est la vitesse 3D totale, tandis que `Vz` est la composante avant/arrière. Le filtre de Kalman réduit le bruit de profondeur avant de calculer cette vitesse.

## Jeu de données à utiliser

Le projet est conçu et testé pour **KITTI Raw Data**, et plus précisément une séquence `synced+rectified data`.

Téléchargez sur le site KITTI, pour la séquence voulue :

1. **synced+rectified data** : obligatoire. Contient `image_02/data`, `image_03/data` et `oxts/data` ;
2. **calibration** : obligatoire. Contient `calib_cam_to_cam.txt` ;
3. **tracklets** : recommandé. Contient `tracklet_labels.xml` pour analyser et valider les objets, mais il n'est pas nécessaire à l'exécution ;
4. **unsynced+unrectified data** : inutile pour ce projet ; ne le téléchargez pas.

**Ne choisissez pas KITTI Object Detection pour le calcul de vitesse.** Il contient des images stéréo indépendantes, mais pas la séquence OXTS nécessaire au suivi temporel utilisé ici.

Exemple de structure attendue :

```text
C:\BIG DATA\PFE\KITTI\
└── 2011_09_26\
    ├── calib_cam_to_cam.txt
    └── 2011_09_26_drive_0005_sync\
        ├── image_02\data\0000000000.png
        ├── image_03\data\0000000000.png
        ├── oxts\data\0000000000.txt
        └── tracklet_labels.xml                 (recommandé)
```

## Installation

Prérequis : Python 3.10 à 3.12 recommandé.

```powershell
cd "C:\BIG DATA\PFE\kitti_stereo_yolo_speed"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Le modèle par défaut est **YOLO11m** : meilleur compromis précision/temps pour les petits objets et les classes proches. Au premier lancement, Ultralytics télécharge automatiquement `yolo11m.pt`. Avec un GPU NVIDIA puissant, vous pouvez remplacer ce modèle par `yolo11l.pt`.

Si votre PC est lent ou ne possède pas de GPU, changez dans `config/default.yaml` :

```yaml
model: yolo11n.pt
```

## Lancer le pipeline complet

```powershell
python scripts\run_sequence.py `
  --left-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0059_sync\image_02\data" `
  --right-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0059_sync\image_03\data" `
  --oxts-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0059_sync\oxts\data" `
  --calib "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_cam_to_cam.txt" `
  --calib-imu-to-velo "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_imu_to_velo.txt" `
  --calib-velo-to-cam "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_velo_to_cam.txt" `
  --tracklets "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0059_sync\tracklet_labels.xml" `
  --max-frames 30 `
  --output-dir "outputs\sequence_0059_zupt"
```

Utilisez `--max-frames 0` pour traiter toute la séquence. Commencez toujours par 30 images afin de vérifier l'installation.

Les résultats sont enregistrés dans :

```text
outputs\sequence_botsort\
├── 0000000000.png                image annotée
├── measurements.csv               mesures complètes
├── speed_track_ID.png             graphique de vitesse
└── distance_track_ID.png          graphique de distance
```

## Créer les graphiques

Après l'exécution :

```powershell
python scripts\plot_results.py `
  --csv "outputs\sequence_botsort\measurements.csv" `
  --track-id 3
```

Omettez `--track-id 3` pour que le programme choisisse automatiquement l'ID avec le plus de vitesses acceptées.

## Validation avec les annotations KITTI

Si `tracklet_labels.xml` et `calib_velo_to_cam.txt` sont disponibles, ajoutez-les au lancement. Le projet projette alors les positions 3D de vérité terrain dans l'image gauche et compare la profondeur et la vitesse relative estimées.

```powershell
python scripts\run_sequence.py `
  --left-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0005_sync\image_02\data" `
  --right-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0005_sync\image_03\data" `
  --oxts-dir "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0005_sync\oxts\data" `
  --calib "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_cam_to_cam.txt" `
  --tracklets "C:\BIG DATA\PFE\KITTI\2011_09_26\2011_09_26_drive_0005_sync\tracklet_labels.xml" `
  --calib-imu-to-velo "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_imu_to_velo.txt" `
  --calib-velo-to-cam "C:\BIG DATA\PFE\KITTI\2011_09_26\calib_velo_to_cam.txt" `
  --max-frames 30

python scripts\report_validation.py `
  --csv "outputs\sequence_botsort\measurements.csv"
```

Le rapport affiche l'erreur absolue moyenne et la RMSE de profondeur, puis l'erreur de vitesse relative. Les vitesses monde ne sont pas validées directement : elles incluent la compensation OXTS de la voiture.

## Filtre objets routiers mobiles

Le fichier `config/default.yaml` définit une zone routière approximative et une vitesse monde minimale positive de `3 km/h`. Avec `require_forward_motion: true`, les objets quasi immobiles ou dont la vitesse monde est négative sont exclus des graphiques et de la sélection automatique. Ils restent dans le CSV brut pour préserver la traçabilité. Désactivez cette option si vous souhaitez analyser les véhicules venant en sens inverse.

## Interpréter les résultats

- `distance_m` : distance devant la caméra, en mètres ;
- `relative_speed_km_h` : vitesse de l'objet par rapport à la voiture KITTI ;
- `speed_3d_km_h` : norme de la vitesse monde 3D obtenue après Kalman et compensation OXTS ;
- `world_vx_mps`, `world_vy_mps`, `world_vz_mps` : composantes du vecteur vitesse monde ;
- `world_speed_km_h` : composante longitudinale `Vz` en km/h ;
- `speed_reliability` : `high`, `medium`, `rejected` ou `waiting` ;
- `waiting` : le même objet n'a pas encore été observé sur assez de frames ;
- `rejected` : mesure écartée parce que la profondeur ou la régression est instable.

Une valeur `high` garantit une bonne cohérence temporelle de la profondeur. Elle n'est pas une vérité terrain absolue : la profondeur stéréo devient moins précise pour les objets lointains, petits ou partiellement masqués.

## Interface Streamlit

```powershell
streamlit run app.py
```

L'interface sert à vérifier une seule paire d'images, sa disparité et sa profondeur. Le traitement professionnel des séquences, le tracking et la vitesse sont réalisés par `scripts/run_sequence.py`.

## Vérifications

```powershell
python -m pytest tests -q
python scripts\demo_synthetic.py
```

## Limites connues à présenter dans le rapport

- Le vecteur vitesse est calculé dans le repère caméra courant. Une vitesse dans un repère monde GPS parfaitement fixe demanderait encore d'intégrer la pose complète (latitude, longitude, orientation) entre les frames.
- YOLO peut confondre certaines classes. La stabilisation par vote sur l'historique ByteTrack réduit ces changements d'étiquette.
- Une personne et un vélo peuvent être fusionnés en `cyclist` lorsqu'ils correspondent au même objet physique.
- Les résultats les plus sensibles sont ceux d'objets lointains, très petits, occultés ou situés sur les bords de l'image.
