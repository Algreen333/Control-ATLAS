EN PRIMER LLOC CAL INSTAL·LAR ARDUPILOT I GAZEBO. VEIEU [GUIA_INSTALL.md](GUIA_INSTALL.md).

# Com simular
Iniciar Gazebo:
```bash
gz sim -v 4 -r <world>.sdf
```
En cas que no estigui disponible la interfaç gràfica i servidor alhora, caldrà executar ``gz sim -v 4 -s -r <world>.sdf`` i en una altra terminal ``gz sim -g``.

Executar SITL:

Poseu per exemple 1 en \<frame\>
```bash
sim_vehicle.py -v ArduCopter -f <frame> --model JSON --map --console
```

Iniciar la càmera:
```bash
gz topic -t "<topic>" -m gz.msgs.Boolean -p "data: 1"
```
On el topic es pot trobar amb `gz topic -l | grep enable_streaming`

Per poder utilitzar el gimbal, cal executar el SITL amb: `sim_vehicle.py -v ArduCopter --model JSON --map --console --add-param-file=<UBICACIO_DEL_ARDUPILOT-GAZEBO>/config/gazebo-iris-gimbal.parm`

Per fer que la càmara apunti cap avall, a la consola d'ardupilot, s'ha de fer els següents comandaments:
```
rc 6 1500
rc 7 1300
rc 8 1500
```

## Exemple de ús: Mapa amb Arucos
Cal executar en terminals diferents:
- `gz sim -v 4 -s -r runway_w_arucos.sdf`. Cal que `runway_w_arucos.sdf` estigui en alguna carpeta configurada al `$GZ_SIM_RESOURCE_PATH` (veure [GUIA_INSTALL.md](GUIA_INSTALL.md))
- `gz sim -g`
- `sim_vehicle.py -v ArduCopter --model JSON --map --console --add-param-file=<UBICACIO_DEL_ARDUPILOT-GAZEBO>/config/gazebo-iris-gimbal.parm`

A continuació s'ha d'inicialitzar la càmera: 
```bash
gz topic -t /world/custom_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data:1"
```
moure-la cap avall (en la terminal de ardupilot creada per sim_vehicle.py ):
```
rc 6 1500
rc 7 1300
rc 8 1500
```

I ja es pot utilitzar qualsevol qualsevol script the python amb ardupilot. 

Per exemple: `codi/gotoaruco.py`. Aquest script busca arucos utilitzant la càmera simulada del Gazebo i, en detectar-los, es posiciona a sobre i baixa fins aterrar. Perquè funcioni primer caldrà situar el dron en una posició on es vegi algun aruco amb la càmera. Les instruccions d'ús són les següents: 
- Executeu `py gotoaruco.py` des d'una terminal.
- A continuació s'obrirà una finestra amb la càmera simulada del gazebo.
- Premeu `s` (en la terminal de gotoaruco.py) per activar/desactivar el mode de pilot autònom. Cal que el dron estigui en mode `guided` (ho podeu fer fàcilment si utilitzeu el Mission Planner).
- També té una opció per gravar vídeos del que captura la càmera. Premeu `r` per gravar o deixar de gravar. Els vídeos es guarden a la ruta des d'on s'esta executant el script amb el nom `output_i.mp4` (output_0.mp4, output_1.mp4, ...) per cada vídeo que es grava. Cada cop que es prem `r` es grava un vídeo nou. Si el programa s'interromp hi ha la possibilitat que es perdi el vídeo.
- Premeu `q` per finalitzar l'execució del programa.

Cal que l'entorn en el que s'executa tingui totes les llibreries instal·lades. (veure [GUIA_INSTALL.md](GUIA_INSTALL.md))

<p align="center">
<img src="https://github.com/Algreen333/Control-ATLAS/blob/main/recursos/gifs/demo_gotoaruco.gif" width="49.5%"/> 
</p>

## Documentació del Gazebo

### Com fer vehicles:
Documentació SDFormat: https://sdformat.org/spec/

### Sensors
SRC: https://gazebosim.org/docs/latest/sensors/
