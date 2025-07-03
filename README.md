# Subspace method based on neural networks (SNN) 

Codes associated with the manuscript titled "Subspace method based on neural networks for solving the partial differential equation" authored by Zhaodong Xu and Zhiqiang Sheng. This repository provides a **modular and user-friendly** implementation, developed with **PyTorch 2.0.0**, and designed for ease of use and modification. In this version, we provide an example for solving the one-dimensional Helmholtz equation; more examples will be provided in future versions.

---

# Abstract

We present a subspace method based on neural networks for solving partial differential equations with high accuracy. The basic idea of our method is to use functions based on neural networks as basis functions to span a subspace, then find an approximate solution in this subspace. Our method uniquely combines the advantages of machine learning and traditional numerical methods. Machine learning methods are utilized to provide the basis functions, while traditional methods efficiently find approximate solutions. Our method can solve various problems, including linear problems, nonlinear problems, and interface problems. Our method can achieve high accuracy with a low cost of training. Numerical examples show that the cost of training these basis functions is low, and only one hundred to two thousand epochs are needed for most tests. The error of our method can even fall below the level of $10^{-10}$ for some tests. The performance of our method significantly surpasses the performance of PINN and DGM in terms of accuracy and computational cost.

---

# Citation 
Zhaodong Xu and Zhiqiang Sheng.
*Subspace method based on neural networks for solving the partial differential equation.* arXiv preprint arXiv:2404.08223, 2024.

**BibTex:**
```
@article{xu2024subspace,
  title={Subspace method based on neural networks for solving the partial differential equation},
  author={Xu, Zhaodong and Sheng, Zhiqiang},
  journal={arXiv preprint arXiv:2404.08223},
  year={2024}
}
```

---

# Paper Acceptance Information

We are pleased to announce that our manuscript, "Subspace method based on neural networks for solving the partial differential equation" by Zhaodong Xu and Zhiqiang Sheng, has been **accepted for publication** in **Computers and Mathematics with Applications**.

* **Journal:** Computers and Mathematics with Applications
* **Acceptance Date:** July 1, 2025

Further details, including the DOI and final publication information, will be updated here once available.


