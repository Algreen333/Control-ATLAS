

### Instal·lar ardupilot
source: https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux

````bash
git clone --recurse-submodules https://github.com/Ardupilot/ardupilot
cd ardupilot
./Tools/environment_install/install-prereqs-<SISTEMA-OPERATIU>.sh -y
````
Cal reobrir la terminal un cop instal·lat.


----
### Instal·lar Gazebo
source: https://gazebosim.org/docs/all/getstarted/

Cal instal·lar Gazebo Harmonic!

#### Ubuntu:
Prerequisits:
````bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg
````
Instal·lació:
```` bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
````


#### Mac:
```` bash
brew tap osrf/simulation
brew install gz-harmonic
````

### NOTA IMPORTANT!
Per poder utilitzar la càmera del gazebo amb `OpenCV`, cal tenir el paquet de GSTREAMER activat. Es pot comprovar si està instal·lat fent:

````python
import cv2
print(cv2.getBuildInformation())
````
i comprovant que apareix `GStreamer: YES`

COM INSTAL·LAR EN CAS QUE NO APAREIXI:
En primer lloc s'ha d'instal·lar GStreamer en cas que no ho estigui: https://gstreamer.freedesktop.org/documentation/installing/index.html?gi-language=c

A continuació s'ha d'instal·lar `OpenCV` amb el paquet activat. Per fer-ho s'ha de compilar a partir del codi font. Tot el procés pot arribar a trigar 30 minuts o més. Primer desinstal·leu `OpenCV` si ja el teniu instal·lat fent `pip uninstall opencv-python`.

```bash
git clone --recurse-submodules https://github.com/opencv/opencv-python.git # Clona el repositori d'opencv-python
cd opencv-python

export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1

# Habilita la instal·lació amb GStreamer
export CMAKE_ARGS="-DWITH_GSTREAMER=ON"

pip install --upgrade pip wheel

# Finalment instal·la OpenCV. 
# Només cal executar aquestes dues línies si ja s'ha fet els passos anteriors i es vol tornar a instal·lar el paquet per un altre entorn o es vol tornar a instal·lar.
python3 -m pip wheel . --verbose
python3 -m pip install opencv_python*.whl
```

Més info a: https://discuss.bluerobotics.com/t/opencv-python-with-gstreamer-backend/8842


### Instal·lar ardupilot-gazebo
```bash
git clone https://github.com/ArduPilot/ardupilot_gazebo

export GZ_VERSION=harmonic
cd ardupilot_gazebo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j<NUM_CORES> # A més cores més ràpid anirà, però consumirà més recursos. Cal que sigui menor que el nombre de cores totals de l'ordinador. ex: 'make -j4'

# Cal modificar la ruta si s'ha clonat el repo a un altre directori. També cal canviar .bashrc per .zshrc si es treballa en mac. 
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/gz_ws/src/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/gz_ws/src/ardupilot_gazebo/models:$HOME/gz_ws/src/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc 

# També es poden afegir més carpetes si es vol guardar els mapes i models a una altra carpeta:
echo 'export GZ_SIM_RESOURCE_PATH=/ruta/a/la/carpeta:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc 

# Per exemple, es podria utilitzar:
echo 'export GZ_SIM_RESOURCE_PATH=<on estigui el github clonat>/assets-gazebo/models:<on estigui el github clonat>/assets-gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
# per tal de poder utilitzat els models i worlds personalitzats de 'assets-gazebo'.
```
Més info a: https://ardupilot.org/dev/docs/sitl-with-gazebo.html

### Llibreries i creació entorn python
Si s'han seguit els passos previs correctament, només caldria instal·lar les llibreries que es troben a `requirements.txt` fent: `pip install -r requirements.txt`. Les llibreries `pymavlink` i `opencv-python` no estan incloses a `requirements.txt`.

Una altra opció és crear un entorn virtual. Això pot portar alguna complicació per les llibreries pròpies d'ardupilot. De totes maneres es pot fer amb 'conda' o 'venv'. Es recomana utilitzar Python 3.10.8.

## Altres codis desenvolupats
### assets-gazebo/generador_arucos.py
Aquest codi serveix per a generar models d'arucos que poden ser utilitzats din del gazebo.  S'utilitza fent:
```bash
py assets-gazebo/generador_arucos.py --id ID_ARUCO --size MIDA_EN_METRES --x POSX --y POSY --z POSZ
```


---
Guia general bastant bona:
https://medium.com/@sanjana_dev9/how-to-set-up-ardupilot-sitl-with-gazebo-for-drone-simulation-a0d15e19b8e3
