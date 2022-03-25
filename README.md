# Velvet

Velvet is a physics engine dedicated for cloth simulation.

## Dependencies

Velvet is built using `C++ 17` and `CUDA 11.1`. Dependencies can be installed using `vcpkg`

```
$ ./vcpkg.exe install {package_name}:x64-windows
```

{package_name} includes:

* glfw3: For window managing

* glad: For OpenGL

* fmt: For better printing functionality

* glm: For 3D mathematics

* assimp: For model loading

* imgui: For graphical user interface (Some extra components are required)

  ```
   ./vcpkg.exe install imgui[core,opengl3-binding,glfw-binding]:x64-windows
  ```
