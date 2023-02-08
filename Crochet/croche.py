import numpy as np
import cv2

def mosaic(img_file, num_colors, block_size, new_name):
    # carrega a imagem original
    img = cv2.imread(img_file)


    # quantiza a imagem para o número desejado de cores
    arr = np.uint8(np.round(img / 256.0) * (256 / num_colors)) 

    # redimensiona a imagem para o tamanho do quadrado desejado
    arr = arr[::block_size, ::block_size]

    # amplia a imagem para o tamanho original
    new_arr = np.repeat(np.repeat(arr, block_size, axis=0), block_size, axis=1)
    cv2.imwrite(new_name+".jpg", new_arr)


    # encontra o tamanho da imagem
    h, w = img.shape[:2] 

    # cria a imagem mosaic
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)

    # preenche a imagem mosaic com as cores médias de cada bloco

    for i in range(0,h,block_size):
        for j in range(0,w,block_size):
            block = img[i:i + block_size, j:j + block_size]
            avg_color = np.mean(np.mean(block, axis=0), axis=0)
            mosaic_img[i:i + block_size, j:j+block_size] = avg_color

    cv2.imwrite(new_name+"2.jpg", mosaic_img)

    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,num_colors,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()].reshape((img.shape))
    res3 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape
    img = np.uint8(np.round(img / (256 // num_colors)) * (256 // num_colors))
    
    mosaic_img = np.zeros(((block_size + 2) * (height // block_size), 
                          (block_size + 2) * (width // block_size), 3), dtype=np.uint8)
    for i in range(height // block_size):
        for j in range(width // block_size):
            block = img[i * block_size: (i + 1) * block_size, 
                        j * block_size: (j + 1) * block_size, :]
            avg_color = np.mean(np.mean(block, axis=0), axis=0)
            mosaic_img[i * (block_size + 2): (i + 1) * (block_size + 2) - 2, 
                       j * (block_size + 2): (j + 1) * (block_size + 2) - 2, :] = avg_color
            mosaic_img[i * (block_size + 2): (i + 1) * (block_size + 2), 
                       j * (block_size + 2) + block_size: (j + 1) * (block_size + 2) + 2, :] = [0, 0, 0]
            mosaic_img[i * (block_size + 2) + block_size: (i + 1) * (block_size + 2) + 2, 
                       j * (block_size + 2): (j + 1) * (block_size + 2), :] = [0, 0, 0]

    
    cv2.imwrite(new_name+"3.jpg", mosaic_img)
    # salva a imagem como um arquivo de imagem


# exemplo de uso
print("  ____                _          _   __  __                 _")
print(" / ___|_ __ ___   ___| |__   ___| |_|  \/  | ___  ___  __ _(_) ___")
print("| |   | '__/ _ \ / __| '_ \ / _ | __| |\/| |/ _ \/ __|/ _` | |/ __|")
print("| |___| | | (_) | (__| | | |  __| |_| |  | | (_) \__ | (_| | | (__")
print(" \____|_|  \___/ \___|_| |_|\___|\__|_|  |_|\___/|___/\__,_|_|\___|")
print()
 
nome = input("Nome da img usada: ")
ncor = int(input("Número de cores desejado: "))
bksz = int(input("Tamanho do bloco: "))
novo = input("Nome da saida: ")
mosaic(nome, ncor, bksz, novo)
print(input("Verifique funcionamento "))
