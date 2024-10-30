#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// shapes
static const int train_size = 60000;
static const int test_size = 10000;
static const int x_sample_size = 28 * 28;
static const int input_size = x_sample_size;
static const int output_size = 10;

// offsets
static const int train_x_offset = 0;
static const int train_y_offset = train_x_offset + (train_size * x_sample_size);
static const int test_x_offset = train_y_offset + train_size * sizeof(uint8_t);
static const int test_y_offset = test_x_offset + (test_size * x_sample_size);

// hyperparameters
static const int epochs = 15;
static const float learning_rate = 0.005;

// random float between 0 and 1.
float randfloat() { return (float)rand() / (float)RAND_MAX; }

// one-hot encoding. y has to be the same size as labels.
void one_hot(int x, int labels, float *y) {
  for (int i = 0; i < labels; i++)
    y[i] = 0;

  y[x] = 1;
}

// softmax activation function. z and a has to be the same size as output_size.
void softmax(float *z, float *a) {
  float exp_z[output_size];
  float sum = 0;

  for (int i = 0; i < output_size; i++) {
    exp_z[i] = exp((double)z[i]);
    sum += exp_z[i];
  }

  for (int i = 0; i < output_size; i++)
    a[i] = exp_z[i] / sum;
}

/* categorical crossentropy loss function. y_true and y_pred has to be the same
 * size as output_size. */
float cce(float *y_true, float *y_pred) {
  float y_pred_clipped[output_size];
  for (int i = 0; i < output_size; i++)
    if (y_pred[i] < 1e-15)
      y_pred_clipped[i] = 1e-15;
    else if (y_pred[i] > (1 - 1e-15))
      y_pred_clipped[i] = 1 - 1e-15;
    else
      y_pred_clipped[i] = y_pred[i];

  float cross_entropy = 0.0;
  for (int i = 0; i < output_size; i++)
    cross_entropy += y_true[i] * log(y_pred_clipped[i]);

  return -cross_entropy;
}

/* derivation of the categorical crossentropy loss function. y_true, y_pred, and
 * d_output has to be the same size as output_size. */
void d_cce(float *y_true, float *y_pred, float *d_output) {
  for (int i = 0; i < output_size; i++)
    if (y_pred[i] < 1e-15)
      d_output[i] = 1e-15 - y_true[i];
    else if (y_pred[i] > (1 - 1e-15))
      d_output[i] = (1 - 1e-15) - y_true[i];
    else
      d_output[i] = y_pred[i] - y_true[i];
}

// argmax, values have to be the same size as size.
int argmax(float *values, int size) {
  int index = 0;

  for (int i = 1; i < size; i++)
    if (values[i] > values[index])
      index = i;

  return index;
}

// infers the model on some data. output has to be the same size as output_size.
void predict(float **weights, float *bias, float *x, float *output) {
  float z[output_size];

  for (int i = 0; i < output_size; i++) {
    float dot_product = 0.0;

    for (int j = 0; j < input_size; j++)
      dot_product += weights[i][j] * x[j];

    z[i] = dot_product + bias[i];
  }

  softmax(z, output);
}

// trains the model based on given weights/bias on epochs, learning rate, and
// data.
void train(float **weights, float *bias, float **x_train, uint8_t *y_train,
           float learning_rate, int epochs) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    float loss = 0.0;
    float accuracy = 1e-15;

    for (int sample = 0; sample < train_size; sample++) {
      float *x = x_train[sample];
      float y = y_train[sample];
      float y_true[output_size];
      one_hot(y, output_size, y_true);

      float y_pred[output_size];
      predict(weights, bias, x, y_pred);
      loss += cce(y_true, y_pred);
      accuracy += argmax(y_pred, output_size) == y;

      for (int i = 0; i < output_size; i++) {
        float error = y_pred[i] - y_true[i];

        for (int j = 0; j < input_size; j++)
          weights[i][j] -= learning_rate * (error * x[j]);

        bias[i] -= learning_rate * error;
      }
    }

    printf("Epoch %d | loss: %f | accuracy: %f\n", epoch + 1, loss / train_size,
           accuracy / train_size);
  }
}

// evaluates the model based on given weights/bias and data.
void evaluate(float **weights, float *bias, float **x_test, uint8_t *y_test,
              float *val_accuracy, float *val_loss) {
  *val_accuracy = 1e-15;
  *val_loss = 0.0;

  for (int sample = 0; sample < test_size; sample++) {
    float *x = x_test[sample];
    float y = y_test[sample];
    float y_true[output_size];
    one_hot(y, output_size, y_true);

    float y_pred[output_size];
    predict(weights, bias, x, y_pred);
    *val_loss += cce(y_true, y_pred);
    *val_accuracy += argmax(y_pred, output_size) == y;
  }

  *val_accuracy /= test_size;
  *val_loss /= test_size;
}

// WARNING: this will seek back to the beginning of the file.
long get_file_size(FILE *file) {
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  return size;
}

int main(void) {
  // seed the rng
  srand(time(NULL));

  // load dataset
  FILE *dataset_file = fopen("datasets/mnist.bin", "rb");
  if (dataset_file == NULL) {
    perror("Couldn't open datasets/mnist.bin");
    return 1;
  }
  size_t dataset_size = (size_t)get_file_size(dataset_file);

  uint8_t *dataset_buffer = malloc(dataset_size * sizeof(uint8_t));
  fread(dataset_buffer, 1, dataset_size, dataset_file);
  fclose(dataset_file);

  float **x_train = malloc(train_size * sizeof(float *));
  for (int i = 0; i < train_size; i++)
    x_train[i] = malloc(x_sample_size * sizeof(float));

  uint8_t *y_train = malloc(train_size * sizeof(uint8_t));

  float **x_test = malloc(test_size * sizeof(float *));
  for (int i = 0; i < test_size; i++)
    x_test[i] = malloc(x_sample_size * sizeof(float));

  uint8_t *y_test = malloc(test_size * sizeof(uint8_t));

  for (int i = 0; i < train_size; i++) {
    const int start = train_x_offset + (i * x_sample_size);
    for (int j = 0; j < x_sample_size; j++)
      x_train[i][j] = (float)dataset_buffer[start + j] / 255.0;
  }

  for (int i = 0; i < train_size; i++)
    y_train[i] = dataset_buffer[train_y_offset + i];

  for (int i = 0; i < test_size; i++) {
    const int start = test_x_offset + (i * x_sample_size);
    for (int j = 0; j < x_sample_size; j++)
      x_test[i][j] = (float)dataset_buffer[start + j] / 255.0;
  }

  for (int i = 0; i < test_size; i++)
    y_test[i] = dataset_buffer[test_y_offset + i];

  // initialize model
  float **weights = malloc(output_size * sizeof(float *));
  float *bias = malloc(output_size * sizeof(float));

  for (int i = 0; i < output_size; i++) {
    weights[i] = malloc(input_size * sizeof(float));
    for (int j = 0; j < input_size; j++)
      weights[i][j] = randfloat() - 0.5;

    bias[i] = 0;
  }

  // training
  train(weights, bias, x_train, y_train, learning_rate, epochs);

  // evaluating
  float val_accuracy, val_loss;
  evaluate(weights, bias, x_test, y_test, &val_accuracy, &val_loss);
  printf("val_accuracy: %f\nval_loss: %f\n", val_accuracy, val_loss);

  // end
  for (int i = 0; i < train_size; i++)
    free(x_train[i]);
  free(x_train);
  free(y_train);

  for (int i = 0; i < test_size; i++)
    free(x_test[i]);
  free(x_test);
  free(y_test);

  for (int i = 0; i < output_size; i++)
    free(weights[i]);
  free(weights);
  free(bias);
  free(dataset_buffer);

  return 0;
}
