#! /bin/bash

# Build docker image
docker build -t pdr-backend-btc_binance:latest .

# Push docker image to Azure Container Registry
az acr login --name opfbot
docker tag pdr-backend-btc_binance:latest opfbot.azurecr.io/pdr-backend-btc_binance:latest
docker push opfbot.azurecr.io/pdr-backend-btc_binance:latest

# Run docker image in Azure Container Instance
az container create --resource-group OPFGroup --name opfbot --image opfbot.azurecr.io/pdr-backend-btc_binance:latest --cpu 1 --memory 1
