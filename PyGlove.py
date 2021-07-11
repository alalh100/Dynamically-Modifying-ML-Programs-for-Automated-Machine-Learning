from random import choice
from tensorflow.keras import datasets, layers, models

# Search space saves all the possible choices
# OBJECTS_PARENTS help us identify which arguments belongs to which object
SEARCH_SPACE = []
OBJECTS_PARENTS = {}


# Add the choices to the search space
def oneof(candidates, hints=None):
    SEARCH_SPACE.append(candidates)
    return candidates


# Abstraction of the normal objects that can save its arguments in a list
# The symbolic object help to recreate the original object with different arguments
class symbolic_object:
    def __init__(self):
        self.args = []
        self.name = None
        self.activation = None
        self.__name__ = 'symbolic'

    # This method save the relationship between the object (parent) and the arguments
    # This helps matching the the object with its arguments
    def __call__(self, *args, activation=None):
        self.args = args

        for i in range(len(self.args)):
            OBJECTS_PARENTS[id(self.args[i])] = id(self)

        if activation is not None:
            self.activation = activation
            OBJECTS_PARENTS[id(activation)] = id(self)
        return self

    def set_name(self, name):
        self.name = name

# Takes a class as an argument and returns a symbolic object
def symbolize(real_object):
    name = real_object.__name__
    sy_obj = symbolic_object()
    sy_obj.set_name(name)
    return sy_obj


# takes a class name, the wanted arguments and the number of arguments and returns an executable object
# Helpful method for the materialize method
def create_object_from_symbols(class_name, args_num, arguments):
    temp_dict = {}
    if args_num == 0:
        code = "temp_dict['var']" + ' = ' + class_name + '()'

    elif args_num == 1:
        arguments = str(arguments[0])
        code = "temp_dict['var']" + ' = ' + class_name + '(' + arguments + ')'

    else:
        args = str(arguments[0])
        for i in range(1, args_num):
            string_arg = str(arguments[i])
            args += ', ' + string_arg
        code = "temp_dict['var']" + ' = ' + class_name + '(' + args + ')'
    exec(code)
    return temp_dict['var']


# totally random 
# returns indices of the choices
def random_search():
    choices = [None] * len(SEARCH_SPACE)
    number_of_choices = len(SEARCH_SPACE)
    for i in range(number_of_choices):
        indices = range(len(SEARCH_SPACE[i]))
        choices[i] = choice(indices)
    return choices


# the search space does not have a complex construction. Its just a collection of choices.
# takes the choices of the search algorithm and return the index of the selected model and how many layers it has.
# helpful for the materialize method
def organized_choices(choices):
    number_of_choices = len(choices) - 1
    # last choice ist the model choice
    num_of_model_chosen = choices[number_of_choices]
    # how many layers for this model
    num_layers = len(SEARCH_SPACE[number_of_choices][num_of_model_chosen])
    return num_of_model_chosen, num_layers


# from an abstract child program to concrete child program based on the search space
# returns an executable model out of the choices of the search algorithm
def materialize(choices):
    number_of_choices = len(SEARCH_SPACE) - 1
    num_of_model_chosen, num_layers = organized_choices(choices)

    # get the layers for this model from the search space    
    layers = []
    old_args = SEARCH_SPACE[number_of_choices][choices[number_of_choices]]

    # recreate the layers
    for i in range(num_layers):
        # if the layer is symbolized
        if 'symbolic' in str(type(old_args[i])):
            symbolic_obj = old_args[i]
            # arg_list = symbolic_obj.get_args()
            selected_args = []
            # get the choice for this list
            for j in range(len(SEARCH_SPACE)):
                if id(SEARCH_SPACE[j]) in OBJECTS_PARENTS:
                    if OBJECTS_PARENTS[id(SEARCH_SPACE[j])] == id(symbolic_obj):
                        if isinstance(SEARCH_SPACE[j][choices[j]], str):
                            activation = 'activation = "' + SEARCH_SPACE[j][choices[j]] + '" '
                            selected_args.append(activation)
                        else:
                            selected_args.append(SEARCH_SPACE[j][choices[j]])
            # add layer. to name
            name = 'layers.' + symbolic_obj.name
            obj = create_object_from_symbols(name, len(selected_args), selected_args)
            layers.append(obj)
        else:
            layers.append(old_args[i])
    # for now just for the Sequential model
    return models.Sequential(layers)


# samples models out of the search space using the choices of the search algorithm
# max_trails is the number of wanted samples 
def sample(trainer, search_algorithm, max_trails=10):
    models = []
    trainer()
    for i in range(max_trails):
        choices = search_algorithm()
        model = materialize(choices)
        models.append(model)
    # Emptying out the search space for reducibility
    SEARCH_SPACE.clear()
    OBJECTS_PARENTS.clear()
    return models
