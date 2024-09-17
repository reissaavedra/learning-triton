import triton
import torch
import triton.language as tl

#Implementing add method using Triton

@triton.jit
def add_kernel(x_ptr: int, y_ptr: int, z_ptr: int, n_elements: int, BLOCK_SIZE: tl.constexpr) -> None:
    """
    A Triton kernel that performs element-wise addition of two tensors.

    Args:
        x_ptr (tl.pointer): Pointer to the first input tensor.
        y_ptr (tl.pointer): Pointer to the second input tensor.
        z_ptr (tl.pointer): Pointer to the output tensor where the result will be stored.
        n_elements (int): The number of elements to process.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed in parallel.

    Returns:
        None: This function does not return a value; it writes the result directly to the output tensor.
    """
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(z_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise addition of two CUDA tensors using a Triton kernel.

    Args:
        x (torch.Tensor): The first input tensor, must be on the CUDA device.
        y (torch.Tensor): The second input tensor, must be on the CUDA device.

    Returns:
        torch.Tensor: A tensor containing the element-wise sum of the input tensors, 
                      also on the CUDA device.

    Raises:
        AssertionError: If the input tensors or the output tensor are not on the CUDA device.
    """
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, debug=True)

    return output


def main() -> None:
    """
    Main function to demonstrate element-wise addition of two CUDA tensors.

    This function initializes two random tensors on the CUDA device, 
    performs addition using a custom Triton kernel, and prints the results.
    
    Returns:
        None: This function does not return a value; it prints the output directly.
    """
    torch.manual_seed(0)
    size: int = 5

    x: torch.Tensor = torch.rand(size, device='cuda')
    y: torch.Tensor = torch.rand(size, device='cuda')

    output: torch.Tensor = add(x, y)
    print(output)
    print(x + y)

    print(type(x.data_ptr()))

main()