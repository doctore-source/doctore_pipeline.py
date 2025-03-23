#!/bin/bash

echo "Updating and Installing Dependencies..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install python3-pip -y

echo "Installing Python Packages..."
pip3 install -r requirements.txt

echo "Starting Gunicorn API Server..."
sudo systemctl daemon-reload
sudo systemctl enable doctore_api.service
sudo systemctl start doctore_api.service

echo "Deployment Completed Successfully."