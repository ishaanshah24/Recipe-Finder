#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
from bs4 import BeautifulSoup



st.write("""
# Recipe Finder
## Upload an image to find the most popular recipes of the food
Once you upload an image, we will identify it and help you find recipes for it.
(As the model is still improving, you may have to upload a different image for it to be correctly identified)""")


from keras.models import load_model
model = load_model('model_vgg19.h5')



def check_space (name):
    if(" " in name):
        name = name.replace(" ","%20")
    return(name)
def GetandParseURL(final_url):
    result = requests.get(final_url)
    soup = BeautifulSoup(result.text,'html.parser')
    return(soup)

def url_finder_epi(food):
    input = check_space(food)
    base_url = 'http://www.epicurious.com/search/'
    food_item = input
    search_url = "?content=recipe&sort=mostReviewed"
    final_url = '{}{}{}'.format(base_url,food_item,search_url)
    soup = GetandParseURL(final_url)
    recipe_links = []
    best_recipes_url = []
    for link in soup.find_all('a'):
        link = link.get('href')
        if link is not None:
            link = link.split(" ")
            recipe_links.append(link)
    u,ind = np.unique(recipe_links, return_index = True)
    best_recipes = u[np.argsort(ind)].tolist()[1:6]
    if any(isinstance(i, list) for i in best_recipes):
        best_recipes = sum(best_recipes,[])
    for element in best_recipes:
        best_recipes_url.append("http://www.epicurious.com"+element)
    recipe_names = []
    for link in soup.find_all('a'):
        link = link.get('aria-label')
        if link is not None:
            recipe_names.append(link)
    u,ind = np.unique(recipe_names, return_index = True)
    best_recipes_names = u[np.argsort(ind)].tolist()[0:5]
    output_df = pd.DataFrame( data = {'Names' :best_recipes_names, 'Links' : best_recipes_url})
    return(output_df)
def url_finder_fn(food):
    input = check_space(food)
    base_url = 'http://www.foodnetwork.com/search/'
    food_item = input
    search_url = "/CUSTOM_FACET:RECIPE_FACET"
    final_url = '{}{}-{}'.format(base_url,food_item,search_url)
    soup = GetandParseURL(final_url)
    recipe_links = []
    for link in soup.find_all('h3', class_ = "m-MediaBlock__a-Headline"):
        link = link.find('a')
        link = link.get("href")[2:]
        recipe_links.append(link)
    best_recipe_links = recipe_links[:5]
    recipe_names = []
    for link in soup.find_all('h3', class_ = "m-MediaBlock__a-Headline"):
        link = link.find(class_ = "m-MediaBlock__a-HeadlineText").text
        if link is not None:
            recipe_names.append(link)
    best_recipe_names = recipe_names[:5]
    output_df = pd.DataFrame( data = {'Names' :best_recipe_names, 'Links' : best_recipe_links})
    return(output_df)
def url_finder_ar(food):
    input = check_space(food)
    base_url = 'https://www.allrecipes.com/search/results/'
    food_item = input
    search_url = "?search"
    final_url = '{}{}={}'.format(base_url,search_url,food_item)
    soup = GetandParseURL(final_url)
    recipe_links = []
    for link in soup.find_all('div',class_ ="card__imageContainer"):
        link = link.find('a')
        link = link.get("href")
        recipe_links.append(link)
    best_recipe_links = recipe_links[:10]
    recipe_names = []
    for link in soup.find_all('div',class_ ="card__imageContainer"):
        link = link.find('a')
        link = link.get("title")
        recipe_names.append(link)
    best_recipe_names = recipe_names[:10]
    output_df = pd.DataFrame( data = {'Names' :best_recipe_names, 'Links' : best_recipe_links})
    return(output_df)



options = st.selectbox("From which website would you like recipes?",
                      ("Epicurious", "Food Network", "All Recipes"))


import keras
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_data = datagen.flow_from_directory('/Users/IshaanShah/Desktop/Recipe_Finder/output/train',
                                        target_size= (224,224), color_mode= "rgb", batch_size = 100,
                                        class_mode= "categorical", shuffle = True, seed = 42)


def prediction(image,model = model):
    image = image.convert('RGB')
    image = image.resize((224,224))
    image = np.asarray(image)/255
    image = tf.expand_dims(image, 0)
    prediction = model.predict(image)
    label_dict = train_data.class_indices
    label = list(label_dict.keys())[list(label_dict.values()).index(np.argmax(prediction))]
    return label


#Upload image
image = st.file_uploader("Upload here",type = ["png","jpg","jpeg"])
if image is not None:
    u_img = Image.open(image)
    st.image(u_img, 'Uploaded Image', use_column_width=True)
    if st.button("Click Here to Classify"):
        with st.spinner('Classifying ...'):
            prediction = prediction(u_img)
            prediction = prediction.replace("_"," ")
            prediction = prediction.title()
        st.write("Algorithm Predicts: "+ prediction)
        if options == "Epicurious":
            st.table(url_finder_epi(prediction))
        if options == "All Recipes":
            st.table(url_finder_ar(prediction))
        if options == "Food Network":
            st.table(url_finder_fn(prediction))
    st.write("Was your result correct?")
    result = st.selectbox("",["","Yes","No"], index = 0)
    if (result=="No"):
        st.write("Use this search box to find recipes")
        food = st.text_input("Type a food item here")
        if (st.button("Search")):
            with st.spinner("Searching..."):
                if options == "Epicurious":
                    st.table(url_finder_epi(food))
                if options == "All Recipes":
                    st.table(url_finder_ar(food))
                if options == "Food Network":
                    st.table(url_finder_fn(food))
    if (result=="Yes"):
        st.write("Thank you!")
