#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import choice
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[5]:


# Search space saves all the possibe choices
# OBJECTS_PARENTS help us identify which arguments belongs to which object
SEARCH_SPACE = []
OBJECTS_PARENTS = {}


# In[11]:


# Add the choices to the search space
def oneof(candidates, hints = None):
    SEARCH_SPACE.append(candidates)
    return candidates


# In[6]:


# Abstraction of the normal objects that can save its arguments in a list
# The symbolic object help to recreate the original object with different arguments
class symbolic_object:
    def __init__(self):
        self.args = []
        self.name = None
        self.activation = None
        
    # This method save the relationship between the object (parent) and the arguments
    def __call__(self, *args, activation = None):
        self.args = args
            
        for i in range(len(self.args)):
            OBJECTS_PARENTS[id(self.args[i])]= id(self)
            
        if(activation is not None):
            self.activation = activation
            OBJECTS_PARENTS[id(activation)] = id(self)
            
        return self
    
    def set_name(self, name):
        self.name = name
        self.__name__ = 'symbolic'
        
    def read_name(self):
        print(self.name)
        
    def get_args(self):
        arg_list = []
        for i in range(len(self.args)):
            arg_list.append(self.args[i])
        
        if(self.activation is not None):
            arg_list.append(self.activation)
        return arg_list


# In[3]:


# Takes a class as an agrument and returns a symbolic object 
def symbolize(real_object):
    name = real_object.__name__   
    sy_obj = symbolic_object()
    sy_obj.set_name(name)
    return sy_obj


# In[4]:


# takes a class name, the wanted agruments and the number of arguments and returns an executable object
# Helpful method for the materlize method
def create_object_from_symbols(class_name, arguments, args_num):
    temp_dict = {}
    if (args_num == 0):
        mycode = "temp_dict['var']"+ ' = ' + class_name+'()'
        
    elif (args_num == 1):
        arguments = str(arguments[0])
        mycode = "temp_dict['var']"+ ' = ' + class_name+'('+arguments+')'
    
    else:
        args = str(arguments[0])
        for i in range(1, args_num):
            arguments[i] = str(arguments[i])
            args += ', ' +arguments[i]
        mycode = "temp_dict['var']"+ ' = ' + class_name+'('+args+')'
    exec(mycode)
    return temp_dict['var']


# In[7]:


# totally random 
# returns indices of the choices
def random_search():
    choices = [None] * len(SEARCH_SPACE)
    number_of_choices = len(SEARCH_SPACE)
    for i in range(number_of_choices):
        indices = range(len(SEARCH_SPACE[i]))
        choices[i] = choice(indices)
    return choices


# In[8]:


# the search space does not have a complex constuction. Its just a collection of choices.
# takes the choices of the search algorithm and return the index of the selected model and how many layers it has.
# helpful for the materlize method
def organized_choices(choices):

    number_of_choices = len(choices)-1
    
    #last choice ist the model choice
    num_of_model_chosen = choices[number_of_choices]
        
    # how many layers for this model
    num_layers = len(SEARCH_SPACE[number_of_choices][num_of_model_chosen])
    
    # number of model
    num_models = len(SEARCH_SPACE[number_of_choices])  
    
    return num_of_model_chosen, num_layers


# In[9]:


# from an abstract child program to concrete child program based on the search space
# returns an executable model out of the choices of the search algorithm
def materialize(choices):
    number_of_choices = len(SEARCH_SPACE)-1
    
    num_of_model_chosen, num_layers = organized_choices(choices)

    # get the layers for this model from the search space    
    args = []
    old_args = SEARCH_SPACE[number_of_choices][choices[number_of_choices]]
    
    for i in range(num_layers):
        #recreate the layers
        if('symbolic' in str(type(old_args[i]))):
            symbolic_obj = old_args[i]
            arg_list = symbolic_obj.get_args()
            selected_args = []
            # get the choice for this list
            for i in range(len(SEARCH_SPACE)):
                if(id(SEARCH_SPACE[i]) in OBJECTS_PARENTS):
                    if(OBJECTS_PARENTS[id(SEARCH_SPACE[i])] == id(symbolic_obj)):
                        if(isinstance(SEARCH_SPACE[i][choices[i]], str)):
                            activation = 'activation = "' + SEARCH_SPACE[i][choices[i]] + '" '
                            selected_args.append(activation)
                        else:
                            selected_args.append(SEARCH_SPACE[i][choices[i]])             
            # add layer. to name 
            name = 'layers.'+symbolic_obj.name
            obj = create_object_from_symbols(name, selected_args, len(selected_args))
            args.append(obj)
        else:
            args.append(old_args[i])
    #for now just for the Sequential model
    return models.Sequential(args)


# In[10]:


# samples models aout of the search space using the choices of the search algorithm
# max_trails is the number of wanted samples 
def sample(trainer, search_algorithm, max_trails = 10):
    models = []
    train = trainer()
    for i in range(max_trails):
        choices = search_algorithm()
        model = materialize(choices)
        models.append(model)
    return models


# In[ ]:




