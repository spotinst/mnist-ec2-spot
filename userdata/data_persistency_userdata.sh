#!/bin/bash
EC2_INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
EC2_AVAIL_ZONE=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
EC2_REGION="`echo \"$EC2_AVAIL_ZONE\" | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"
DEVICE_NAME="/dev/xvdb"
DATA_STATE="unknown"

# Validate that volumes are ready
until [ "$DATA_STATE" == "attached" ]; do
        DATA_STATE=$(aws ec2 describe-volumes \
                --region $EC2_REGION \
                --filters \
                Name=attachment.instance-id,Values=$EC2_INSTANCE_ID \
                Name=attachment.device,Values=$DEVICE_NAME \
                --query Volumes[].Attachments[].State \
                --output text)
        echo $DATA_STATE
        sleep 5
done
# Volumes are ready

# Maount volumes
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
cd ~/ec2-spot-labs/scripts/

# Download dataset if not already downloaded before
[ "$(ls -A /dl/datasets/)" ] && echo "Not Empty" || curl -o /dl/datasets/mnist.npz https://s3.amazonaws.com/img-datasets/mnist.npz
# Start anacunda ENV and run the training script
sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate tensorflow_p27; python train_network.py "


