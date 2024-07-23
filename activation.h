#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Sigmoid activation function
double sigmoid(double x);

// Derivative of Sigmoid activation function
double sigmoid_d(double x);

// ReLU activation function
double relu(double x);

// Derivative of ReLU activation function
double relu_d(double x);

// Derivative of tanh activation function
double tanh_d(double x);

#endif
