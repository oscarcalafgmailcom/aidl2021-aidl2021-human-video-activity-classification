# A la Deep Learning VM
conda -c pytorch install pytorch torchvision torchaudio cudatoolkit
# però està donant un error de merda de que no troba tochvision.ioy

# actualitzo conda que m'ho demana a veure si és això
conda update -n base conda

# Scripts de la cuda 

sudo nano /etc/apt/sources.list

deb http://deb.debian.org/debian/ buster main contrib non-free
deb-src http://deb.debian.org/debian/ buster main contrib non-free
deb http://security.debian.org/debian-security buster/updates main contrib non-free
deb-src http://security.debian.org/debian-security buster/updates main contrib non-free
deb http://deb.debian.org/debian/ buster-updates main contrib non-free
deb-src http://deb.debian.org/debian/ buster-updates main contrib non-free

sudo apt update
sudo apt install nvidia-detect
sudo nvidia-detect

# A partir d'aquí no ha funcionat
# sudo apt install nvidia-driver
# sudo reboot
# fins aquí

# Amb el resultat del nvida detect vas a https://www.nvidia.com/en-us/drivers/unix/ i mires quin driver t'has de baixar. En aquest cas Latest Legacy GPU version (390.xx series): 390.143
sudo apt install wget
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/390.143/NVIDIA-Linux-x86_64-390.143.run
sudo apt -y install linux-headers-$(uname -r) build-essential libglvnd-dev pkg-config
# En el meu cas el nom de l'arxiu ha canviat
# sudo echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf
sudo echo blacklist nouveau > /etc/modprobe.d/nvidia-blacklists-nouveau.conf

sudo bash NVIDIA-Linux-x86_64-390.143.run 

# Hauria de funcionar
nvidida-smi

# Install git
sudo apt install git

# Install cuda
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda


sudo apt install curl
conda install pytorch torchvision torchaudio cudatoolkit

wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-debian10-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo dpkg -i cuda-repo-debian10-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-debian10-11-4-local/7fa2af80.pub
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda

nvidia-smi

conda install -c anaconda cudatoolkit


He creat una debian amb dos discos
i intento executar l'scripts de la cuda
això peta, així que et fa instalar coses abans, és anar buscant a internet fins que aconsegueixes que s'instali. 



# Això és per intentar arreglar la puta màquina
sudo /sbin/rmmod nvidia

# Només cal que facis un dels dos següents, supodadament
sudo /sbin/modprobe nvidia
sudo nvidia-smi

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c 

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch



# Poso al /etc/apt/sources.list
# Debian 9 "Stretch"
deb http://httpredir.debian.org/debian/ stretch main contrib non-free


sudo apt update
sudo apt install linux-headers-$(uname -r|sed 's,[^-]*-[^-]*-,,') nvidia-legacy-340xx-driver


Deep Learning VM
Solución proporcionada por Google Click to Deploy
Instance
k80com-vm
Instance zone
europe-west1-b
Instance machine type
n1-highmem-2
 MÁS ACERCA DEL SOFTWARE
Comenzar con Deep Learning VM
SSH

Próximos pasos sugeridos
Set up the Cloud SDK.
The Cloud SDK (gcloud) is the preferred command line tool for interfacing with your instance. Download it here. 

(Optional) Copy files to your VM from your local machine.
You can use the gcloud tool to upload files to your machine.

$
gcloud compute scp --project amazing-craft-318807 --zone europe-west1-b --recurse <local file or directory> k80com-vm:~/

Access the running Jupyter notebook.
We've already started a Jupyter notebook instance on the VM for your convenience. In order to get link that can be used to access Jupyter Lab run the following command.

$
gcloud compute instances describe --project amazing-craft-318807 --zone europe-west1-b k80com-vm | grep googleusercontent.com | grep datalab

Assign a static external IP address to your VM instance
An ephemeral external IP address has been assigned to the VM instance. If you require a static external IP address, you may promote the address to static. Learn more 

Documentación
Official Documentation 
StackOverflow: Deep Learning VM 
Google Group: Deep Learning VM 
Asistencia
If you have non-framework related issues, you can bring them up at the Deep Learning VM Stack Overflow .

Propiedades de la plantilla
 SHOW MORE
