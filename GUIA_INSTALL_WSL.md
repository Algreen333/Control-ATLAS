## Guia d'instal·lació en un entorn WSL2

Primer s'ha d'instal·lar WSL2 amb la distro de Ubuntu
Instruccions WSL: https://learn.microsoft.com/es-es/windows/wsl/install


Abans de continuar és recomanable asegurar-se que no hi hagi cap instalació prèvia de opencv-python ni opencv-python-headless


### 1r pas: Instal·lar ardupilot
source: https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux

````bash
git clone --recurse-submodules https://github.com/Ardupilot/ardupilot
cd ardupilot
./Tools/environment_install/install-prereqs-<SISTEMA-OPERATIU>.sh -y
````
Cal reobrir la terminal un cop instal·lat.


----
### 2n pas: Instal·lar Gazebo Harmonic
source: https://gazebosim.org/docs/all/getstarted/



````bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg

sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
````
### 3r pas: Instal·lar gstreamer
[Source](https://gstreamer.freedesktop.org/documentation/installing/index.html?gi-language=c)
```bash
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

 ```

### 4t pas: Instal·lar OpenCV-python amb gstreamer
Primer desinstal·leu `OpenCV` si ja el teniu instal·lat fent `pip uninstall opencv-python`.

Com que no existeixen binaris (o almenys no s'han trobat) a internet de OpenCV per a Ubuntu en WSL2 amb el Gstreamer activat, toca compilar-so 
#### (ATENCIÓ, aquest procés pot tardar entre 10-40 min)
#### Primer instal·lar dependències
``` bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly python3-dev python3-numpy
```
#### A continuació descarregar OpenCV i OpenCV contrib
```bash
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.9.0
cd ../opencv_contrib
git checkout 4.9.0
```
Ara precompilem
```bash
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_EXAMPLES=OFF \
      ../CMakeLists.txt
```
Comproveu que apareix en la consola:
```
GStreamer: YES
GStreamer base: YES
```
A continuació compilar
```bash
make -j$(nproc)
```
Finalment instal·lem OpenCV
```bash
sudo make install
sudo ldconfig
```
I ara fem
```python
import cv2
print(cv2.getBuildInformation())
```
Per comprovar que apareixi:
```
GStreamer: YES
```
#### Si no s'ha pogut carregar cv2 perquè esteu en venv-ardupilot podeu fer el següent: 
```bash
ln -s /usr/local/lib/python3.12/site-packages/cv2/ ~/venv-ardupilot/lib/python3.12/site-packages/cv2
```

### 5è pas: Instal·lar ardupilot-gazebo
Aqui podeu decidir si escollir el repo original o el modificat per Rafael ( conté les malles 3D del dron simplificades, només els 5% de polígons, molt recomanat per millorar el rendiment )
```bash
sudo apt-get install rapidjson-dev -y

## Versió Rafael:
git clone https://github.com/ultrazar/ardupilot_gazebo
# Versió original git clone https://github.com/ArduPilot/ardupilot_gazebo

export GZ_VERSION=harmonic
cd ardupilot_gazebo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc) 

# Cal modificar la ruta si s'ha clonat el repo a un altre directori. També cal canviar .bashrc per .zshrc si es treballa en mac. 
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}:$HOME/Control-ATLAS/assets-gazebo/worlds:$HOME/Control-ATLAS/assets-gazebo/models' >> ~/.bashrc 

# També es poden afegir més carpetes si es vol guardar els mapes i models a una altra carpeta:
# echo 'export GZ_SIM_RESOURCE_PATH=/ruta/a/la/carpeta:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc 

# Per exemple, es podria utilitzar:
# echo 'export GZ_SIM_RESOURCE_PATH=<on estigui el github clonat>/assets-gazebo/models:<on estigui el github clonat>/assets-gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
# per tal de poder utilitzar els models i worlds personalitzats de 'assets-gazebo'.
```
Més info a: https://ardupilot.org/dev/docs/sitl-with-gazebo.html

### 6è pas: Descarregar aquest mateix repositori
Això és una mica obvi però heu de descarregar aquest repo :\)

```bash
cd ~/
git clone https://github.com/Algreen333/Control-ATLAS.git
```

### Últims pasos
Fins aquí ja tendriem tots els programes instal·lats, ho podeu provar amb `gz sim -v 4  fast_runway_w_arucos.sdf`

El que pot passar és que és molt probable que el rendiment sigui molt pobre degut a que la renderització del gazebo s'està duent a terme per CPU i no per GPU. Això és perquè per defecte en WSL2 s'utilitza el driver de pantalla llvmpipe que processa les imatges per CPU per assegurar la compatibilitat.  Podeu buscar per internet per si teniu sort i aconseguiu fer-ho funcionar amb la vostra GPU: pistes que he trobat -->
```bash
# per saber quin tipus de renderer feu servir podeu executar:
glxinfo -B
# Si apareix llvmpipe esteu fent servir processat per CPU
# Podeu buscar per internet o preguntar-li al ChatGPT com podeu utilitzar la vostra GPU per processat d'imatge dintre de WSL2...
```
També és molt recomanable que instal·leu el Mission Planner en el vostre Windows (podeu conectar-vos al dron simulat localment en WSL2 per connexió UDP) 
[Mission Planner](https://ardupilot.org/planner/docs/mission-planner-installation.html)



