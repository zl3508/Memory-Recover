import os
import sys
import signal
import time
from edge_impulse_linux.audio import AudioImpulseRunner

# Parameter
MENU_MODEL_PATH = "./model_menu.eim"
YESNO_MODEL_PATH = "/model_yesno"
THRESHOLD = 0.65

runner = None

def signal_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def wait_for_wake_word(model_select="menu", device_id=None):
    global runner

    if model_select == "menu":
        model_file = MENU_MODEL_PATH
    elif model_select == "yesno":
        model_file = YESNO_MODEL_PATH
    else:
        raise ValueError("model_select must be 'menu' or 'yesno'.")

    modelfile = model_file

    with AudioImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')

            selected_device_id = 2
            if device_id is not None:
                selected_device_id = int(device_id)
                print("Device ID " + str(selected_device_id) + " has been provided as an argument.")

            for res, audio in runner.classifier(device_id=selected_device_id):
                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                for label in labels:
                    score = res['result']['classification'][label]
                    print('%s: %.2f\t' % (label, score), end='')
                # maxium class and threshold
                results = res['result']['classification']
                best_label = max(results, key=results.get)
                best_score = results[best_label]

                if best_score >= THRESHOLD:
                    if model_select == "menu" and (best_label == "takephoto" or best_label == "himan"):
                        return best_label
                    elif model_select == "yesno" and (best_label == "yes" or best_label == "no"):
                        return best_label
                print('', flush=True)


        finally:
            if runner:
                runner.stop()

# for test
if __name__ == '__main__':
    label = wait_for_wake_word(model_select="menu")
    print(f"Detected label: {label}")