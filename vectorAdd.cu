{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "### Assignment 2\n",
        "Below is an example that runs native CUDA code.\n",
        "\n",
        "1.   We investigate the CUDA version, drivers and the avaiable GPU with nvidia-smi and nvcc-version\n",
        "2.   We use the IPython magic command \"%%writefile filename\" to save a *.cu program\n",
        "3.   We then compile and run the *.cu program with nvcc\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "q0-ZomlNSrF5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc --version\n",
        "!nvidia-smi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yJlrROAn5Rqj",
        "outputId": "8b52cb4b-594a-473d-c4a3-2cac3d25523b"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nvcc: NVIDIA (R) Cuda compiler driver\n",
            "Copyright (c) 2005-2023 NVIDIA Corporation\n",
            "Built on Tue_Aug_15_22:02:13_PDT_2023\n",
            "Cuda compilation tools, release 12.2, V12.2.140\n",
            "Build cuda_12.2.r12.2/compiler.33191640_0\n",
            "Wed Nov 13 09:19:16 2024       \n",
            "+---------------------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |\n",
            "|-----------------------------------------+----------------------+----------------------+\n",
            "| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                                         |                      |               MIG M. |\n",
            "|=========================================+======================+======================|\n",
            "|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   66C    P8              11W /  70W |      0MiB / 15360MiB |      0%      Default |\n",
            "|                                         |                      |                  N/A |\n",
            "+-----------------------------------------+----------------------+----------------------+\n",
            "                                                                                         \n",
            "+---------------------------------------------------------------------------------------+\n",
            "| Processes:                                                                            |\n",
            "|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |\n",
            "|        ID   ID                                                             Usage      |\n",
            "|=======================================================================================|\n",
            "|  No running processes found                                                           |\n",
            "+---------------------------------------------------------------------------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "## Next, we write a native CUDA code and save it as 'vectorAdd.cu'\n"
      ],
      "metadata": {
        "id": "vsbr4brQH6v3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile vectorAdd.cu\n",
        "#include <stdio.h>\n",
        "#include <stdlib.h>\n",
        "__global__ void add(int *a, int *b, int *c) {\n",
        "*c = *a + *b;\n",
        "}\n",
        "int main() {\n",
        "int a, b, c;\n",
        "// host copies of variables a, b & c\n",
        "int *d_a, *d_b, *d_c;\n",
        "// device copies of variables a, b & c\n",
        "int size = sizeof(int);\n",
        "// Allocate space for device copies of a, b, c\n",
        "cudaMalloc((void **)&d_a, size);\n",
        "cudaMalloc((void **)&d_b, size);\n",
        "cudaMalloc((void **)&d_c, size);\n",
        "// Setup input values\n",
        "c = 0;\n",
        "a = 3;\n",
        "b = 5;\n",
        "// Copy inputs to device\n",
        "cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);\n",
        "  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);\n",
        "// Launch add() kernel on GPU\n",
        "add<<<1,1>>>(d_a, d_b, d_c);\n",
        "// Copy result back to host\n",
        "cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);\n",
        "  if(err!=cudaSuccess) {\n",
        "      printf(\"CUDA error copying to Host: %s\\n\", cudaGetErrorString(err));\n",
        "  }\n",
        "printf(\"result is %d\\n\",c);\n",
        "// Cleanup\n",
        "cudaFree(d_a);\n",
        "cudaFree(d_b);\n",
        "cudaFree(d_c);\n",
        "return 0;\n",
        "}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T0S5AUrl4eI8",
        "outputId": "d846423a-3a6d-4003-a87f-58f586efbc77"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting vectorAdd.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## We compile the saved cuda code using nvcc compiler"
      ],
      "metadata": {
        "id": "TqdaBa9wIICn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!nvcc vectorAdd.cu -o vectorAdd\n",
        "!ls\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TvYps9NQ40NC",
        "outputId": "098c9c4e-4f43-4131-c8a7-4a528ba91856"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "a.out  sample_data  vectorAdd  vectorAdd.cu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Finally, we execute the binary of the compiled code"
      ],
      "metadata": {
        "id": "SUALHJy9IPvG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!./vectorAdd"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gF-iISqy7O2E",
        "outputId": "52abd7e2-bc1b-4c56-9bcc-e2faf31674c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "result is 8\n"
          ]
        }
      ]
    }
  ]
}