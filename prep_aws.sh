# this script installs all dependencies required to run the bot

echo "Installing Apex-GO dependencies..."

sudo apt-get update
sudo npm install forever -g
sudo apt-get -y install python3-pip

sudo pip3 install tensorflow
sudo pip3 install keras
