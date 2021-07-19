import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import models
import utils
from utils import pprint
import debug

utils.use_GPU()

x_train, x_test = utils.get_data(["mnist", "fashion_mnist"], test=True)
x_train = x_train / 255.0
x_test = x_test / 255.0
image_shape = utils.image_shape_from_data(x_train)

if len(x_train) == 0:
    pprint("No data exists.", 'fail')
    exit()

debug_mode = False

if debug_mode:
    debug.run()
    x_train = None
    x_test = None
else:
    pprint("0. Exit", 'blue')
    pprint("1. Create new basic model", 'blue')
    pprint("2. Create new basic mirror model", 'blue')
    pprint("3. Create new custom model", 'blue')
    pprint("4. Create new custom mirror model", 'blue')
    pprint("5. Load existing model", 'blue')
    choice = input("Choice: ")
    model = None

    if choice == "0" or choice == "":
        exit()
    elif choice == "5":
        model = utils.load_model_options()
        pprint("0. Exit \n1. Don't train loaded model \n2. Train loaded model", 'blue')
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
            keyword = "custom"
        elif choice == "4":
            keyword = "custom_mirror"
        else:
            pprint("Wrong choice.", "red")
            exit()
        model = models.create_model(keyword, image_shape, "", "", 0, [], [], 'SSIM', inp=True)
        model = utils.test_train_model(model, x_train, inp=True)

    encoder = utils.get_encoder(model)
    decoder = utils.get_decoder(model)

    for _ in range(5):
        test = utils.random_test(x_test)
        encoded = encoder.predict(test.reshape(-1, image_shape[0], image_shape[1], image_shape[2]))[0]
        decoded = decoder.predict(encoded.reshape(-1, len(encoded)))[0]
        utils.plot2(test, decoded)

    print()
    pprint(f"Original Message Length: {utils.number_of_pixels(utils.image_shape(test))}", "HEADER")
    pprint(f"Encoded Message Length: {len(encoded)}", "HEADER")
    pprint(f"Compression Ratio: {utils.number_of_pixels(utils.image_shape(test)) / len(encoded)}", "HEADER")

