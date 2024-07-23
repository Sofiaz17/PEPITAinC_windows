#include "get_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
// #define N_SAMPLES 60000
// #define N_DIMS 784
// #define N_CLASSES 10
// #define N_TEST_SAMPLES 10000


// Function to read training and test data and store them appropriately
void read_csv_file(float** data, float* label_i, float** lbl_vec, char* dataset){
    printf("in read_csv_file\n");
    FILE *file;
    if(strcmp(dataset, "train") == 0){
       // printf("dataset = train\n");
        file = fopen("mnist_train.csv", "r");
    }
    else if(strcmp(dataset, "test") == 0){
        file = fopen("mnist_test.csv", "r");
    }
    if(file == NULL){
        printf("Error reading the file!");
        exit(1);
    }
    char buffer[3200];  //784 pixel * (3 char + 1 comma) + 1 label char + 1 comma = 3138 -> one line of csv file
    int i = 0;
    //printf("before while\n");
    while(fgets(buffer, sizeof(buffer), file)){     //legge da file tanti 3200 char alla volta e salva in buffer
        char* token = strtok(buffer, ",");  //saves in array 'token' first value of buffer ( ',' is delimitator)
        int j = 0;
        while(token != NULL){
            if(j == 0){            
                label_i[i] = (float)atoi(token);  //transforms char in int (primo token, label)
            }
            else{
                data[i][j-1] = (float)atoi(token);    //salva valori pixel in matrice (ma non il primo che è la label)
            }
            j++;
            token = strtok(NULL, ",");    //continua a processare i token della riga letta
        }
        i++;
    }
   // printf("after while\n");
    fclose(file);
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;      //60000
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES; //10000
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_CLASSES;j++){
            if((int)label_i[i] == j){    //se label i-esima tra tutte quelle del file == j
                lbl_vec[i][j] = 1.0;          //y di sample i class j (target.one_hot)
            }
            else{
                lbl_vec[i][j] = 0.0;
            }
        }
    }
    ///printf("end of read_csv_file\n");
}

// Function to scale the dataset
void scale_data(float** data, char* dataset){
    printf("in scale_data\n");
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES;
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_DIMS;j++){  //N_DIMS=784
            data[i][j] = (float)data[i][j]/(float)255.0;   //scale between 0 and 1 every pixel
        }
    }
}

void fetch_batch(float** batch_data, float** batch_labels, float batch_size, int batch_index, float** img, float** lbl){
    int i;
    printf("in fetch batch\n");
    for (i = 0; i < batch_size; i++) {
        int data_index = batch_index * batch_size + i;
        if (data_index >= N_SAMPLES){
            printf("There are no more samples!\n");
            break;  // Prevent out-of-bounds access
        }

        for (int j = 0; j < N_DIMS; j++) {
            batch_data[i][j] = img[data_index][j];
            
        }
        for(int j=0;j<N_CLASSES;j++){
            batch_labels[i][j] = lbl[data_index][j];
        }

    }
    //batch_index += batch_index + batch_size;
    printf("end of fetch_batch\n");
}

// Function to normalize the dataset
void normalize_data(float** X_train, float** X_test){    //X_train -> data (pixel values)
    printf("in normalize_data\n");
    float* mean = malloc(N_DIMS*sizeof(float));
    float total = N_SAMPLES;   //60000
    for(int i=0;i<N_DIMS;i++){
        float sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += X_train[j][i];
        }
        mean[i] = sum/total;    //media su ogni dimensione (sommando tutti i sample)
    }
    float* sd = malloc(N_DIMS*sizeof(float));
    for(int i=0;i<N_DIMS;i++){ //784
        float sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += pow(X_train[j][i] - mean[i], 2);     //standard deviation
        }
        sd[i] = sqrt(sum/total);
    }
    for(int i=0;i<N_DIMS;i++){
        for(int j=0;j<N_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_train[j][i] = (float)(X_train[j][i] - mean[i])/(float)sd[i];
            }
        }
        for(int j=0;j<N_TEST_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_test[j][i] = (float)(X_test[j][i] - mean[i])/(float)sd[i];
            }
        }
    }
    free(sd);
    free(mean);
    printf("end of normalize\n");
}