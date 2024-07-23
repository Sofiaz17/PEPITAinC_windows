#include "activation.h"
#include "get_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define NUM_LAYERS 3
#define NUM_NEU_L1 784
#define NUM_NEU_L2 128
#define NUM_NEU_L3 10
#define BATCH_SIZE 64
#define EPOCHS 5

int neu_per_lay[] = {NUM_NEU_L1, NUM_NEU_L2, NUM_NEU_L3};

const float mom_gamma = 0.9;
int batch_index = 0;
int exp_number = 0;

void matrix_multiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    // Initialize the result matrix C to zero
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0.0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < A_rows; i++) {
        for (int k = 0; k < A_cols; k++) {
            for (int j = 0; j < B_cols; j++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void matrix_transpose(float* A, float* B, int rows, int cols) {
    // A is rows x cols
    // B will be cols x rows (transpose of A)

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_subtract(float* A, float* B, float* C, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i * cols + j] = A[i * cols + j] - B[i * cols + j];
        }
    }
}

typedef struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    float*** w;
    float** b;
    float*** momentum_w;
    //float*** momentum2_w;
    float** momentum_b;
    //float** momentum2_b;
    float** error;
    float** actv_in;
    float** actv_out;
    float* targets;
}NeuralNet;

void initialize_net(NeuralNet* nn){
    int i,j,k;

    if(nn->n_layers == 0){
        printf("No layers in Neural Network...\n");
        return;
    }

    printf("Initializing net...\n");

    for(i=0;i<nn->n_layers-1;i++){

        for(j=0;j<nn->n_neurons_per_layer[i];j++){          //££

            for(k=0;k<nn->n_neurons_per_layer[i+1];k++){     //££

                float nin = nn->n_neurons_per_layer[i+1];
                float limit = sqrtf(6.0f/nin);
                float scale = rand() / (float) RAND_MAX;

                //CHECK VALUE
                // Initialize Output Weights for each neuron
                nn->w[i][j][k] = -limit + scale * (limit + limit);
                nn->momentum_w[i][j][k] = 0.0;
                //printf("w[%d][%d][%d] defined \n",i,j,k);
            }
            if(i>0){
                nn->b[i][j] = 0.0;
                nn->momentum_b[i][j] = 0.0;
                //printf("bias[%d][%d]\n",i,j);
            }
        }
    }
    
    for (j=0; j<nn->n_neurons_per_layer[nn->n_layers-1]; j++){
        //printf("nn->n_neurons_per_layer[nn->n_layers-1]: %d\n",nn->n_neurons_per_layer[nn->n_layers-1]);
        //printf("nn->n_layers-1: %d\n",nn->n_layers-1);
        nn->b[nn->n_layers-1][j] = 0.0;
        //printf("bias[n_layers-1][%d]\n",nn->n_layers-1,j);
    }

    // for(int i=0;i<nn->n_layers;i++){
    //     for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££
    //         //printf("actv_in = 0.0\n");
    //         nn->actv_in[i][j] = 0.0;
    //     }
    // }

    printf("net initialized\n");

}

void free_NN(NeuralNet* nn);


// Function to create a neural network and allocate memory
NeuralNet* newNet(){
    printf("in newNet\n");
    int i,j;
    //space for nn
    NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = NUM_LAYERS;
    //space for layers
    nn->n_neurons_per_layer = (int*)malloc(nn->n_layers * sizeof(int));

    //initialize layer with num neurons
    for(i=0; i<nn->n_layers; i++){
        nn->n_neurons_per_layer[i] = neu_per_lay[i];
    }

    //space for weight matrix and weight momentum (first dimension)->layer
    nn->w = (float***)malloc((nn->n_layers-1)*sizeof(float**));
    nn->momentum_w = (float***)malloc((nn->n_layers-1)*sizeof(float**));
    //space for bias matrix and bias momentum (first dimension)->layer
    nn->b = (float**)malloc((nn->n_layers-1)*sizeof(float*));
    nn->momentum_b = (float**)malloc((nn->n_layers-1)*sizeof(float*));
    

    for(int i=0;i<nn->n_layers-1;i++){
        //weight matrix and momentum (second dimension)->neurons of curr layer
        nn->w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));   //+1?? (per tutti in for)  //££
        nn->momentum_w[i] = (float**)malloc((nn->n_neurons_per_layer[i])*sizeof(float*));                   //££
        //bias matrix and mometum (second dimension)->neurons of curr layer
    
        nn->b[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));                    //££
        nn->momentum_b[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));                    //££
        
        //space for weight matrix and weight momentum (third dimension)->neurond of next layer
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){  //+1??
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));     //+1??             //££
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(float));                    //££
        }
    }

    nn->b[nn->n_layers-1] =  (float*)malloc((nn->n_neurons_per_layer[nn->n_layers-1])*sizeof(float));
    
    //space for error matrix for each neuron in each layer(layer dimension) 
    nn->error = (float**)malloc((nn->n_layers)*sizeof(float*));
    //space for input and output to activation functions (layer dimension)
    nn->actv_in = (float**)malloc((nn->n_layers)*sizeof(float*));
    nn->actv_out = (float**)malloc((nn->n_layers)*sizeof(float*));
    
    for(int i=0;i<nn->n_layers;i++){
        //space for error matrix for each neuron in each layer(neuron dimension) 
        nn->error[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float)); //+1??                    //££
        //space for input and output to activation functions (neuron dimension)
        nn->actv_in[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));                    //££
        nn->actv_out[i] = (float*)malloc((nn->n_neurons_per_layer[i])*sizeof(float));                    //££
    }
    //space for desired outputs (one hot vector)
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1])*sizeof(float));    //+1??                    //££
    
    // Initialize the weights
    initialize_net(nn);

    printf("end newNet\n");
    //return SUCCESS_CREATE_ARCHITECTURE;
    return nn;
}

// Function to free the dynamically allocated memory
void free_NN(struct NeuralNet* nn){
    printf("in free_NN\n");
    if(!nn) return;
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
           // free(nn->momentum2_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
       // free(nn->momentum2_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
        //free(nn->momentum2_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    //free(nn->momentum2_w);
    free(nn->b);
    free(nn->momentum_b);
    //free(nn->momentum2_b);
    for(int i=0;i<nn->n_layers;i++){
        free(nn->actv_in[i]);
        free(nn->actv_out[i]);
        free(nn->error[i]);
    }
    free(nn->actv_in);
    free(nn->actv_out);
    free(nn->error);
    free(nn->targets);
    free(nn->n_neurons_per_layer);
}


// Function for forward propagation step
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){
    // printf("in forward prop\n");
    //initialize input to actv for every layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££
            //printf("actv_in = 0.0\n");
            nn->actv_in[i][j] = 0.0;
        }
    }
    for(int i=1;i<nn->n_layers;i++){
        // printf("i: %d\n", i);
        // Compute the weighted sum -> add bias to every input
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
            nn->actv_in[i][j] += 1.0 * nn->b[i-1][j];       //why *1.0?
            //printf("actv_in + bias\n");
        }
        //  printf("nn->n_neurons_per_layer[i-1]: %d\n", nn->n_neurons_per_layer[i-1]);
        //  printf("nn->n_neurons_per_layer[i]: %d\n", nn->n_neurons_per_layer[i]);

        //add previous weighted output
        for(int k=1;k<nn->n_neurons_per_layer[i-1];k++){
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){                       //££ + 0
                nn->actv_in[i][j] += nn->actv_out[i-1][k] * nn->w[i-1][k][j];
                //printf("[i:%d][k:%d][j:%d] actv_in con w\n", i,k,j);
                //printf("[i:%d][k:%d][j:%d]\n", i,k,j);
            }
             //printf("[i:%d][k:%d]\n", i,k);
            //printf("[i:%d][k:%d]\n", i,k);
            //printf("nn->n_neurons_per_layer[i-1]\n", nn->n_neurons_per_layer[i-1]);
        }
        // Apply non-linear activation function to the weighted sums
        //if last layer, apply softmax
        if(i == nn->n_layers-1){
            // printf("i==n_layers-1\n");
            if(strcmp(loss, "mse") == 0){
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££ + 0
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0){
                float max_input_to_softmax = (float)INT_MIN;
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                    if(fabs(nn->actv_in[i][j]) > max_input_to_softmax){
                        max_input_to_softmax = fabs(nn->actv_in[i][j]);
                    }
                }
                float deno = 0.0;
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                    nn->actv_in[i][j] /= max_input_to_softmax;
                    deno += exp(nn->actv_in[i][j]);
                }
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                    nn->actv_out[i][j] = (float)exp(nn->actv_in[i][j])/(float)deno;
                    
                }
                // printf("ce loss\n");
            }
        } //if other layers, apply something else
        else{
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){        //+1??                    //££ + 0 
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->actv_out[i][j] = tanh(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                    //printf("relu activation\n");
                }
                else{
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                }
            }
        }
    }
}


// Function to calculate loss
float calc_loss(struct NeuralNet* nn, char* loss){
    // printf("in calc loss\n");
    int i;
    float running_loss = 0.0;
    int last_layer = nn->n_layers-1;
    for(i=0;i<nn->n_neurons_per_layer[last_layer];i++){       //+1???                    //££ + 0
        if(strcmp(loss, "mse") == 0){
            running_loss += (nn->actv_out[last_layer][i] - nn->targets[i]) * (nn->actv_out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            running_loss -= nn->targets[i]*(log(nn->actv_out[last_layer][i]));
        }
	}
    if(strcmp(loss, "mse") == 0){
        running_loss /= BATCH_SIZE;
    }
    return running_loss;
}


void shuffle(int* arr, size_t n) {
    if (n > 1) {
        // Seed the random number generator to ensure different results on each run
        srand(time(NULL));

        for (size_t i = n - 1; i > 0; i--) {
            // Generate a random index j such that 0 <= j <= i
            size_t j = rand() % (i + 1);

            // Swap arr[i] with arr[j]
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
}

// Function to train the model for 1 epoch
void model_train(struct NeuralNet* nn, float** X_train, float** y_train, float* y_train_temp, float** X_test, float** y_test, float* y_test_temp,
                 char* activation_fun, char* loss, char* opt, float learning_rate){
    printf("in model train\n");
    // Create an array for generating random permutation of training sample indices
    // int shuffler_train[N_SAMPLES];    //better to do on whole dataset???
    // for(int i=0;i<N_SAMPLES;i++){
    //     shuffler_train[i] = i;
    // }
    // shuffle(shuffler_train, N_SAMPLES);

    //FIRST FORWARD PASS FOR ACTIVATIONS (dropout mask)
   
    // Start training the model for 1 epoch and simultaneously calculate the training error and accuracy
  
    float* test_accs = (float*)malloc(EPOCHS*sizeof(float));
    float* train_losses = (float*)malloc(EPOCHS*sizeof(float));
    float* train_accs = (float*)malloc(EPOCHS*sizeof(float));
    float curr_loss = 0.0;

    float* B = (float*)malloc(N_CLASSES*N_DIMS*sizeof(float));
    
    int nin = 28*28;
    float sd = sqrtf(6.0f/(float)nin);
    for(int i=0;i<N_CLASSES;i++){
        for(int j=0;j<N_DIMS;j++){
            float rand_num = ((float)rand() / RAND_MAX);
             B[i * N_DIMS + j] = (rand_num * 2 * sd - sd) * 0.05;
        }
    }

    float* outputs = (float*)malloc(BATCH_SIZE*N_CLASSES*sizeof(float));
    float* targets = (float*)malloc(BATCH_SIZE*N_CLASSES*sizeof(float));
    float* inputs = (float*)malloc(BATCH_SIZE*N_DIMS*sizeof(float));
    float** layers_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));     //n_lay-1, batch, neu
    for(int i=0; i<NUM_LAYERS-1;i++){
        layers_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float* error = (float*)malloc(BATCH_SIZE*N_CLASSES*sizeof(float));
    float* error_input = (float*)malloc(BATCH_SIZE*N_DIMS*sizeof(float));
    float* mod_inputs = (float*)malloc(BATCH_SIZE*N_DIMS*sizeof(float));
    float* mod_outputs = (float*)malloc(BATCH_SIZE*N_CLASSES*sizeof(float));
    float** mod_layers_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));     //n_lay-1, batch, neu
    for(int i=0; i<NUM_LAYERS-1;i++){
        mod_layers_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float* mod_error = (float*)malloc(BATCH_SIZE*N_CLASSES*sizeof(float));
    float** delta_w_all = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int l=0; l<NUM_LAYERS-1;l++){
        delta_w_all[l] = (float*)malloc(nn->n_neurons_per_layer[l+1]*nn->n_neurons_per_layer[l]*sizeof(float));
    }
    float* mod_error_T = (float*)malloc(N_CLASSES*BATCH_SIZE*sizeof(float));
    float** delta_lay_act = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act[i] = (float*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(float));
    }
    float** delta_lay_act_T = (float**)malloc((NUM_LAYERS-1)*sizeof(float*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act_T[i] = (float*)malloc(nn->n_neurons_per_layer[i+1]*BATCH_SIZE*sizeof(float));
    }
    printf("memory allocated\n");

    for(int epoch=0;epoch<EPOCHS;epoch++){      //remember not real batch size but tot number of images used (one per time)
        
        int shuffler_train[N_SAMPLES];
        for(int i=0;i<N_SAMPLES;i++){
            shuffler_train[i] = i;
        }

        shuffle(shuffler_train, N_SAMPLES);
   
        if(epoch == 50){
            learning_rate *= 0.2;
        }
        
        float running_loss = 0.0;
        int batch_count = 0;
        float total = 0.0;
        float correct = 0.0;
        
        int idx = -1;
        float max_val = (float)INT_MIN;

        //alloca batch array
        for(int batch_num=0;batch_num<floor(N_SAMPLES/BATCH_SIZE);batch_num++){
            //assegna batch
            printf("[%d]TRAIN BATCH %d\n", epoch, batch_num);
                
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //do masks
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    nn->actv_out[0][j] = X_train[shuffler_train[batch_elem+batch_count*BATCH_SIZE]][j];      //assign input ??????i?????
                }
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    nn->targets[j] = y_train[shuffler_train[batch_elem+batch_count*BATCH_SIZE]][j];        //assign target labels (one hot)
                }
                for(int in_neu=0;in_neu<N_DIMS;in_neu++){
                    inputs[batch_elem*N_DIMS+in_neu] = nn->actv_out[0][in_neu];
                }
               
                forward_propagation(nn, activation_fun, loss);
                //array to save output for every batch
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    outputs[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu]; 
                    targets[batch_elem*N_CLASSES+out_neu] = nn->targets[out_neu];
                }
            }
            total += BATCH_SIZE;
            // printf("batch[%d] first forward pass\n",batch_num);
            for(int lay=0;lay<NUM_LAYERS-1;lay++){
                for(int batch=0;batch<BATCH_SIZE;batch++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        layers_act[lay][batch*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu]; //?????????
                    }
                }
            }
            // printf("batch[%d] lay_act\n",batch_num);
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                for(int j=0;j<N_CLASSES;j++){
                    error[batch_elem*N_CLASSES+j] = outputs[batch_elem*N_CLASSES+j] - nn->targets[j];
                }
            }
            // printf("batch[%d] error\n",batch_num);
            
            // for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
            //     for(int j=0;j<N_CLASSES;j++){
            //         error_input[batch_elem*N_CLASSES+j] =  error[batch_elem*N_CLASSES+j] * B[batch_elem*N_CLASSES+j] 
                     
            matrix_multiply(error, B, error_input, BATCH_SIZE, N_CLASSES, N_DIMS);
            // //matrix multiplication
            // for(int i = 0; i < BATCH_SIZE; i++) {
            //     for(int j = 0; j < N_DIMS; j++) {
            //         error_input[i*N_DIMS+j] = 0.0;
            //         for (int k = 0; k < N_CLASSES; k++) {
            //             error_input[i*N_DIMS+j] += error[i*N_CLASSES+k] * B[k*N_DIMS+j];
            //         }
            //     }
            // }
           
            for(int i = 0; i < BATCH_SIZE; i++) {
                for(int j = 0; j < N_DIMS; j++) {
                    mod_inputs[i*N_DIMS+j] = inputs[i*N_DIMS+j] + error_input[i*N_DIMS+j];
                }
            }
            // printf("batch[%d] mod_inputs\n",batch_num);
           
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //do masks
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    nn->actv_out[0][j] = mod_inputs[batch_elem*nn->n_neurons_per_layer[0] + j];      //assign input
                }
                
                forward_propagation(nn, activation_fun, loss);
                //array to save output for every batch
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    mod_outputs[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu]; 
                }

                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){      //trova neurone con output maggiore
                        max_val = nn->actv_out[nn->n_layers-1][j];
                        idx = j;       //££ + 0 j-1
                    }
                }
                
                if(idx == (int)y_train_temp[shuffler_train[batch_elem+batch_count*BATCH_SIZE]]){   //checks train prediction
                    correct++;
                }
            }
            // printf("batch[%d] second forward pass\n",batch_num);
            
            for(int lay=0;lay<NUM_LAYERS-1;lay++){
                for(int batch=0;batch<BATCH_SIZE;batch++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        mod_layers_act[lay][batch*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu]; //?????????
                    }
                }
            }
            // printf("batch[%d] mod_layers_act\n",batch_num);
            //mod activations
          
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                for(int j=0;j<N_CLASSES;j++){
                    mod_error[batch_elem*N_CLASSES+j] = mod_outputs[batch_elem*N_CLASSES+j] - nn->targets[j];
                }
            }
            // printf("batch[%d] mod_error\n",batch_num);

            for(int row=0;row<N_CLASSES;row++){
                for(int col=0;col<BATCH_SIZE;col++){
                    mod_error_T[row*BATCH_SIZE+col] = mod_error[col*N_CLASSES+row]; //neg
                }
            }
            // printf("batch[%d] mod_error_T\n",batch_num);

            for(int l=0;l<NUM_LAYERS-1;l++){    //LOOOP???????? 
                // neg_delta_lay_act_T[l] = -(layers_act[l] - mod_layers_act[l]);
                matrix_subtract(layers_act[l], mod_layers_act[l], delta_lay_act[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                matrix_transpose(delta_lay_act[l], delta_lay_act_T[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);    //neg
                
                float* delta_w = (float*)malloc(nn->n_neurons_per_layer[l+1]*nn->n_neurons_per_layer[l]*sizeof(float));
                if(l == NUM_LAYERS-1){
                    if((NUM_LAYERS-1) > 1){
                        matrix_multiply(mod_error_T, mod_layers_act[l], delta_w, N_CLASSES, BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                    }
                    else {
                        matrix_multiply(mod_error_T, mod_inputs, delta_w, N_CLASSES, BATCH_SIZE, N_DIMS);
                    }
                } 
                else if(l==0){
                    matrix_multiply(delta_lay_act_T[l], mod_inputs, delta_w, nn->n_neurons_per_layer[l+1], BATCH_SIZE, N_DIMS);
                } 
                else if(l>0 && l<NUM_LAYERS-1){
                    matrix_multiply(delta_lay_act_T[l], mod_layers_act[l-1], delta_w, nn->n_neurons_per_layer[l+1], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                }
                delta_w_all[l] = delta_w;
                free(delta_w);

            }
            // printf("batch[%d] delta computed\n",batch_num);
            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){                    //££ + 0
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){    

                        if(strcmp(opt, "sgd") == 0){
                            nn->w[k][i][j] -= learning_rate * (delta_w_all[k][j*nn->n_neurons_per_layer[k+1]+i]/BATCH_SIZE);  //WHY BATCH_SIZE???
                        }
                        else if(strcmp(opt, "momentum") == 0){
                            nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (1.0-mom_gamma) * (delta_w_all[k][j*nn->n_neurons_per_layer[k+1]+i]/BATCH_SIZE) * learning_rate;
                            nn->w[k][i][j] -= nn->momentum_w[k][i][j];
                        }
                    }
                }
            }
            // printf("batch[%d] w updated\n",batch_num);
            running_loss += calc_loss(nn, loss);  
            batch_count += 1;
        }

        curr_loss = running_loss / (float)batch_count;
        printf("[%d, %5d] loss: %.3f\n", epoch, batch_count, curr_loss);
        train_losses[epoch] = curr_loss;
        printf("Train correct: %.2f\n", correct);
        printf("Train total: %-2f\n", total);
        printf("Train accuracy epoch [%d]: %.4f %%\n", epoch, 100 * correct / total);
        train_accs[epoch] = 100 * correct / total;

        printf("TESTING...\n");

        correct = 0.0;
        total = 0.0;
        batch_count = 0;

        int shuffler_test[N_TEST_SAMPLES];
        for(int i=0;i<N_TEST_SAMPLES;i++){
            shuffler_test[i] = i;
        }

        shuffle(shuffler_test, N_TEST_SAMPLES);

        for(int batch_num=0;batch_num<floor(N_TEST_SAMPLES/BATCH_SIZE);batch_num++){
            // printf("[%d]TEST BATCH %d\n", epoch,batch_num);

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    nn->actv_out[0][j] = X_test[shuffler_test[batch_elem+batch_count*BATCH_SIZE]][j];      //assign input ??????i?????
                }
                
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    nn->targets[j] = y_test[shuffler_test[batch_elem+batch_count*BATCH_SIZE]][j];        //assign target labels (one hot)
                }
                
                forward_propagation(nn, activation_fun, loss);
            
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){
                        max_val =nn->actv_out[nn->n_layers-1][j];
                        idx = j;        //££ + 0 j-1
                    }
                }
                
                
                if(idx == (int)y_test_temp[shuffler_test[batch_elem+batch_count*BATCH_SIZE]]){
                    correct++;
                }
            }
            total += BATCH_SIZE;
            // printf("batch[%d] test finished\n",batch_num);
            batch_count += 1;
        }
        printf("Test accuracy epoch [%d]: %f %%\n",epoch, 100 * correct / total);
        test_accs[epoch] = 100 * correct / total;

        char buf[30];
        snprintf(buf, sizeof(buf), "PEPITA_Cimplementation[epoch:%d].txt", epoch);

        FILE *file = fopen(buf, "w");
        printf("open file\n");
        fprintf(file, "EPOCH %d\n", epoch);
    
        fprintf(file, "Train loss epoch [%d]: %lf\n", train_losses[epoch]);
        fprintf(file, "Train accuracy epoch [%d]: %lf\n", epoch, train_accs[epoch]);
        fprintf(file, "Test accuracy epoch [%d]: %lf\n", epoch, test_accs[epoch]);

        fclose(file);
        printf("close file\n");
        // exp_number += 1;
    }

    
    printf("FINISHED TRAINING\n");

    printf("Freeing memory...\n");
    free(test_accs);
    free(train_losses);
    free(train_accs);
    free(B);
    free(outputs);
    free(targets);
    free(inputs);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(layers_act[i]);
    }
    free(layers_act);
    free(error);
    free(error_input);
    free(mod_inputs);
    free(mod_outputs);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(mod_layers_act[i]);
    }
    free(mod_layers_act);
    free(mod_error);
    for(int l=0; l<NUM_LAYERS-1;l++){
        free(delta_w_all[l]);
    }
    free(delta_w_all);
    free(mod_error_T);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(delta_lay_act_T[i]);
    }
    free(delta_lay_act_T);  
}
 
                 



// // Function to test the model
// float* model_test(struct NeuralNet* nn, float** X_test, float** y_test, float* y_test_temp, char* activation_fun, char* loss){
//     printf("in model test\n");
//     int correct = 0;
//     float running_loss = 0.0;
//     float curr_loss = 0.0;
//     for(int i=0;i<BATCH_SIZE;i++){
//         int idx = -1;
//         float max_val = (float)INT_MIN;
//         for(int j=0;j<nn->n_neurons_per_layer[0];j++){        //££ + 0
//             nn->actv_out[0][j] = X_test[i][j];
//         }
//         for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
//             nn->targets[j] = y_test[i][j];
//         }
//         forward_propagation(nn, activation_fun, loss);
//         running_loss += calc_loss(nn, loss);
            
//         for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
//             if(nn->actv_out[nn->n_layers-1][j] > max_val){
//                 max_val =nn->actv_out[nn->n_layers-1][j];
//                 idx = j;        //££ + 0 j-1
//             }
//         }
//         if(idx == (int)y_test_temp[i]){
//             correct++;
//         }
//     }
//     curr_loss = running_loss / (float)BATCH_SIZE;
//     //running_loss /= (float)N_TEST_SAMPLES;
//     float accuracy = (float)correct*100/(float)BATCH_SIZE;
//     static float metrics[2];
//     metrics[0] = curr_loss;
//     metrics[1] = accuracy;
//     return metrics;
// }


int main(){

    // Used for setting a random seed
    srand(time(NULL));


    // Initialize neural network architecture parameters
    // int n_layers = 3;
    // int n_neurons_per_layer[] = {784, 128, 10};

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet();
    //init_nn(nn);
 
    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    float learning_rate = 0.1;
    float init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "momentum";
    int num_samples_to_train = 64;
    int epochs = 100;

    float** img_train;
    float** lbl_train;
    float* lbl_train_temp;
    float** img_test;
    float** lbl_test;
    float* lbl_test_temp;
    // float** batch_train_data; 
    // float** batch_train_labels;
    // float** batch_test_data; 
    // float** batch_test_labels;
    

    // float* train_losses = (float*)malloc(epochs*sizeof(float));
    // float* train_accuraces = (float*)malloc(epochs*sizeof(float));
    // float* test_losses = (float*)malloc(epochs*sizeof(float));
    // float* test_accuraces = (float*)malloc(epochs*sizeof(float));


    img_train = (float**) malloc(N_SAMPLES*sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        img_train[i] = (float*)malloc(N_DIMS*sizeof(float));
    }
    lbl_train = malloc(N_SAMPLES * sizeof(float*));
    for(int i=0;i<N_SAMPLES;i++){
        lbl_train[i] = malloc(N_CLASSES * sizeof(float));
    }
    lbl_train_temp = malloc(N_SAMPLES*sizeof(float));
    read_csv_file(img_train, lbl_train_temp, lbl_train, "train");
    scale_data(img_train, "train");

    img_test = malloc(N_TEST_SAMPLES*sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        img_test[i] = malloc(N_DIMS*sizeof(float));
    }
    lbl_test = malloc(N_TEST_SAMPLES * sizeof(float*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        lbl_test[i] = malloc(N_CLASSES * sizeof(float));
    }
    lbl_test_temp = malloc(N_TEST_SAMPLES*sizeof(float));
    read_csv_file(img_test, lbl_test_temp, lbl_test, "test");
    printf("heading to scale_data\n");
    scale_data(img_test, "test");
    normalize_data(img_train, img_test);

    model_train(nn,img_train,lbl_train,lbl_train_temp,img_test,lbl_test,lbl_test_temp,activation_fun,loss,opt,learning_rate);

    // batch_train_data = (float**) malloc(BATCH_SIZE*sizeof(float*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_train_data[i] = (float*)malloc(N_DIMS*sizeof(float));
    // }
    // //printf("batch_train_data allocated\n");
    // batch_train_labels = malloc(BATCH_SIZE * sizeof(float*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_train_labels[i] = malloc(N_CLASSES * sizeof(float));
    // }
    // //printf("batch_train_lbl allocated\n");
    // batch_test_data = (float**) malloc(BATCH_SIZE*sizeof(float*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_test_data[i] = (float*)malloc(N_DIMS*sizeof(float));
    // }   
    // // printf("batch_test_data allocated\n");
    // batch_test_labels = malloc(BATCH_SIZE * sizeof(float*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_test_labels[i] = malloc(N_CLASSES * sizeof(float));
    // }
    // // printf("batch_test_lbl allocated\n");
  
    

    // Initialize file to store metrics info for each epoch
    // FILE* file = fopen("BP_C_implementation.txt", "w");
    // printf("file opened\n");
    // fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    // float curr_train_loss;
    // float curr_train_acc;
    // float curr_test_loss;
    // float curr_test_acc;

  

    // // Train the model for given number of epoch and test it after every epoch
    // for(int itr=0;itr<epochs;itr++){
    //     printf("in epoch for\n");
       
    //     int batch_count = 0;
    //     for(int btr=0;btr<floor(N_SAMPLES/BATCH_SIZE);btr++){
    //         printf("in batch for\n");
    //         // fetch_batch(batch_train_data, batch_train_labels, BATCH_SIZE, batch_index, img_train, lbl_train);

    //         // float* train_metrics = model_train(nn, batch_train_data, batch_train_labels, lbl_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
           
           

    //         // curr_train_loss = train_metrics[0] / batch_count;
    //         // curr_train_acc = train_metrics[1] / batch_count; //????????
    //         // train_losses[itr+1] = curr_train_loss;
    //         // train_accuraces[itr+1] = curr_train_acc;

    //         fprintf(file, "%lf,", train_losses[itr+1]);
    //         fprintf(file, "%lf,", train_accuraces[itr+1]);

    //         // printf("TRAIN...\n");
    //         // printf("Epoch, batch count: [%d, %5d] -> ", itr+1,batch_count);
    //         // printf("Train loss: %lf, ", train_losses[itr+1]);
    //         // printf("Train Accuracy: %lf, ", train_accuraces[itr+1]);
    
    //         // learning_rate = init_lr * exp(-0.1 * (itr+1));
    //         // batch_index += BATCH_SIZE;
    //     }
    //     for(int btr=0;btr<floor(N_TEST_SAMPLES/BATCH_SIZE);btr++){
    //         // printf("in batch test for\n");
    //         // fetch_batch(batch_test_data, batch_test_labels, BATCH_SIZE, batch_index, img_test, lbl_test);

        

    //         // float* test_metrics = model_test(nn, img_test, lbl_test, lbl_test_temp, activation_fun, loss);
    //         // float test_loss = test_metrics[0];
    //         // float test_acc = test_metrics[1];

    //         // curr_test_loss = test_metrics[0] / batch_count;
    //         // curr_test_acc = test_metrics[1] / batch_count; //????????
    //         // test_losses[itr+1] = curr_test_loss;
    //         // test_accuraces[itr+1] = curr_test_acc;

    //         fprintf(file, "%lf,", test_losses[itr+1]);
    //         fprintf(file, "%lf\n", test_accuraces[itr+1]);

    //         // printf("TEST...\n");
    //         // printf("Epoch, batch count: [%d, %5d] -> ", itr+1,batch_count);
    //         // printf("Test loss: %lf, ", test_losses[itr+1]);
    //         // printf("Test Accuracy: %lf\n", test_accuraces[itr+1]);

    //         // batch_index += BATCH_SIZE;
    //     }
    // }

    // // Close the file
    // fclose(file);

    // Free the dynamically allocated memory
    free_NN(nn);
    // free(train_losses);
    // free(train_accuraces);
    // free(test_losses);
    // free(test_accuraces);
    
    free(img_train);
    free(lbl_train);
    free(lbl_train_temp);
    for(int i=0;i<N_SAMPLES;i++){
        free(img_train[i]);
        free(lbl_train[i]);
    }
    free(img_test);
    free(lbl_test);
    free(lbl_test_temp);
    for(int i=0;i<N_TEST_SAMPLES;i++){
        free(img_test[i]);
        free(lbl_test[i]);
    }
    return 0;
}