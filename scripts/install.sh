#!/bin/bash
mkdir -p data/processed
curl -L -o data/food-com-recipes-and-user-interactions.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions
cd data
unzip food-com-recipes-and-user-interactions.zip
