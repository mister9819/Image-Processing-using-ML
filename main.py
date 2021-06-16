import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import models
import utils

utils.use_GPU()

x_train, x_test = utils.get_data("mnist", test=True)
x_train = x_train / 255.0
x_test = x_test / 255.0
image_shape = utils.image_shape_from_data(x_train)

choice = "1"
debug = False

if not debug:
    print("0. Exit")
    print("1. Create new basic model")
    print("2. Create new basic mirror model")
    print("3. Create new custom mirror model")
    print("4. Load existing model")
    choice = input("Choice: ")
model = None

if choice == "0" or choice == "":
    exit()
elif choice == "3":
    model = utils.load_model_options()
    print("0. Exit \n1. Don't train loaded model \n2. Train loaded model")
    choice = int(input("Choice: "))
    if choice == 0:
        exit()
    elif choice == 1:
        pass
    elif choice == 2:
        model = utils.train_model(model, x_train)
else:
    keyword = ""
    if choice == "1":
        keyword = "basic"
    elif choice == "2":
        keyword = "basic_mirror"
    elif choice == "3":
        keyword = "custom_mirror"
    else:
        print("Wrong choice.")
        exit()
    model = models.create_model(keyword, image_shape, "C", "C", 64, [28, 28], inp=not debug)
    model = utils.train_model(model, x_train, inp=True)

encoder = utils.get_encoder(model)
decoder = utils.get_decoder(model)

test = utils.random_test(x_test)
encoded = encoder.predict(test.reshape(-1, image_shape[0], image_shape[1], image_shape[2]))[0]
decoded = decoder.predict(encoded.reshape(-1, len(encoded)))[0]
print("Encoded Message:", encoded)
print("Original Message Length", utils.number_of_pixels(utils.image_shape(test)))
print("Encoded Message Length:", len(encoded))
print("Compression Ratio:", len(encoded) / utils.number_of_pixels(utils.image_shape(test)))
utils.plot2(test, decoded)
