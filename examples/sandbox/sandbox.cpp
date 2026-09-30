#include <iostream>
#include <vector>
#include <iomanip>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h" // Modern execution engine header

static void print_tensor(const std::string& name, struct ggml_tensor* tensor) {
    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "Shape: [" << tensor->ne[0] << ", " << tensor->ne[1] << "]" << std::endl;
    
    float* data = (float*)tensor->data;
    for (int r = 0; r < tensor->ne[1]; ++r) {
        std::cout << "  ";
        for (int c = 0; c < tensor->ne[0]; ++c) {
            int idx = r * tensor->ne[0] + c;
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << data[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // 1. Initialize modern context (Metadata storage only, no_alloc = true)
    // In modern GGML, raw memory storage lives on the backend, not inside the context!
    size_t ctx_size = 1024 * 1024; // 1 MB overhead
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // only reserves memory inside the context for tensor metadata
    };
    struct ggml_context* ctx = ggml_init(params);

    // 2. Initialize the CPU Backend Device
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);;
    if (!backend) {
        std::cerr << "Failed to init CPU backend" << std::endl;
        return 1;
    }

    // 3. Define Shapes
    struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2); // num columns (3), num rows (2)
    struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    struct ggml_tensor* bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);

    // 4. Set up an allocator buffer to provision raw storage memory
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    
    // Allocate space for input output tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5. Populate values safely
    float data_A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float data_B[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float data_bias[4] = {-50.0f, -50.0f, -50.0f, -50.0f};

    ggml_backend_tensor_set(A,    data_A,    0, ggml_nbytes(A));
    ggml_backend_tensor_set(B,    data_B,    0, ggml_nbytes(B));
    ggml_backend_tensor_set(bias, data_bias, 0, ggml_nbytes(bias));

    // 6. Chain Graph Operators
    struct ggml_tensor* matmul_res = ggml_mul_mat(ctx, B, A);
    struct ggml_tensor* add_res    = ggml_add(ctx, matmul_res, bias);
    struct ggml_tensor* final_out  = ggml_relu(ctx, add_res);

    // 7. Trace the forward computation graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, final_out);

    // Allocate intermediate scratchpads required by the graph nodes
    // update: only calculates and provisions the internal buffer capacity needed to run a graph. 
    // It does not map that memory to the tensors in your current graph. 
    // All intermediate operation data pointers remain NULL
    // ggml_gallocr_reserve(gallocr, gf);
    ggml_gallocr_alloc_graph(gallocr, gf);

    // 8. Execute the graph via the modern backend runtime
    std::cout << "=== Executing via modern Backend API ===" << std::endl;
    ggml_backend_graph_compute(backend, gf);

    // Read outputs back from backend to host memory to print
    std::vector<float> out_buffer(ggml_nelements(final_out));
    ggml_backend_tensor_get(final_out, out_buffer.data(), 0, ggml_nbytes(final_out));
    final_out->data = out_buffer.data(); // Temporary assignment to utilize print function

    print_tensor("Final Output (After Bias + ReLU)", final_out);

    // Clean up
    ggml_gallocr_free(gallocr);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return 0;
}
