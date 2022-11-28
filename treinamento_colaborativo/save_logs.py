import os

class SaveLoss:
    def __init__(self, dir_data='gdrive/MyDrive/Colab Notebooks/'):
        self.dir_data = dir_data

    def save_loss_in_file(self, history_loss, name='train_loss.txt'):
        # escreve no final do txt
        with open(f'{self.dir_data}{name}', 'a') as f:
            f.write(f"{history_loss},")

    def open_file(self, name='train_loss.txt'):
        txt_file = open(f'{self.dir_data}{name}', "r")
        file_content = txt_file.read()
        list_loss = [x for x in file_content.split(",")]
        txt_file.close()

        list_loss.reverse()

        list_loss = list_loss[1:]
        list_loss = [float(x) for x in list_loss]

        list_loss.reverse()

        return list_loss

    def remove_file(self):
        os.remove(f'{self.dir_data}train_loss.txt')
        os.remove(f'{self.dir_data}valid_loss.txt')
