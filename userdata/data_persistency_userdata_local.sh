#!/bin/bash
EC2_AVAIL_ZONE=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
VOL='/dev/xvdb'
VOLCHECK=`ls /dev/xvdb`

# Validate that VOLumes are ready
until [ $VOLCHECK = $VOL ]; do
  sleep 5
done
# Volumes are ready

# Maount VOLumes
sudo mkdir -p /dl
sudo mkfs -t xfs /dev/xvdb
sudo mount /dev/xvdb /dl
sudo chown -R ubuntu: /dl/

# Prepare datasets and checkpoints folder if not exist
cd /dl
mkdir -p datasets
mkdir -p checkpoints
mkdir -p models

# Get training code from github
mkdir -p ~/ec2-spot-labs
git clone https://github.com/essale/mnist-ec2-spot.git
chown -R ubuntu: mnist-ec2-spot
cd /dl/mnist-ec2-spot/scripts/

# Download dataset if not already downloaded before
[ "$(ls -A /dl/datasets/)" ] && echo "Not Empty" || curl -o /dl/datasets/mnist.npz https://s3.amazonaws.com/img-datasets/mnist.npz
# Start anacunda ENV and run the training script
sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate tensorflow_p27; python train_network.py "