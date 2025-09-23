# Code Style Guide

This document outlines the code style guidelines for the StatGPT project. The guidelines aim to ensure consistency,
readability, and maintainability across the codebase.

---

## 1. Environment Variables

- **Use pydantic-settings**: Leverage the `pydantic-settings` package to manage environment variables and application
  settings.
- **Centralize Configuration**: Place all introduced environment variables (with any default values) within a dedicated
  `<root_module>.settings` package.
- **Document Thoroughly**: Ensure that each environment variable is documented in the package’s README to clarify
  its purpose and default value.

## 2. Encapsulation and Visibility

- **Protected by Default**: Use protected (single underscore, `_`) visibility for class fields and methods by default.
    - Only use public attributes (no underscore) when it is truly necessary for external usage.
    - This approach encourages cautious exposure of class internals and maintains control over the class’s public
      interface.

## 3. Clarity in Field Types

- **Clear Types**: Make class field types obvious to anyone reading the code:
    - Use descriptive variable names.
    - Provide type hints so readers and tools (e.g., IDEs, linters) can easily understand expected types.
    - Consider docstrings if necessary to clarify more complex structures.

## 4. Avoid Static Field Initialization in Class Bodies

- **No Conditional Assignments in Class**:
  ```python
  class A:
    if condition:
      b = ...
    else:
      b = ...
  ```
    - Refrain from putting conditional or complex logic directly in class bodies.
    - Prefer class-level constants (e.g., `CONSTANT_VAR = 42`) or inline definitions if needed.
    - Move logic to an `__init__` method, a factory function, or a dedicated function for clarity.

## 5. Global State and Mutability

- **Avoid Mutable Global State**:
    - Whenever possible, pass mutable objects as parameters rather than storing them globally.
    - If truly necessary, ensure thread-safe access (e.g., locks) and document usage clearly.

## 6. Pythonic Naming and Structure

- **Method and Function Names**: Use `snake_case` for method and function names.
- **Class Names**: Use `PascalCase` (e.g., `MyClass`).
- **Constants**: Use `UPPER_CASE` for constants or immutable values.

## 7. Modern Type Hints

- **Use the correct Built-In Generics**:
  - It’s recommended to use built-in generic types instead of `typing.List`, `typing.Dict`,
    `typing.Tuple`, etc. For example:
      - Use `list[str]` instead of `typing.List[str]`.
      - Use `dict[str, int]` instead of `typing.Dict[str, int]`.
      - Use `tuple[str, float]` instead of `typing.Tuple[str, float]`.
  - Some other generics have been moved from the `typing` module to `collections.abc.`
    It is recommended to import them from the new module. For, example:
      - Use `collections.abc.Iterable` instead of `typing.Iterable`.
      - Use `collections.abc.Iterator` instead of `typing.Iterator`.
      - Use `collections.abc.Callable` instead of `typing.Callable`.
- **Annotations for Readability**:
    - Write function signatures with type hints, e.g., `def my_func(name: str) -> None: ...`.
    - This improves code readability and assists with IDE-based autocompletion.
- **Union Types**:
    - In Python 3.10+, you can use the “pipe” (`|`) symbol to indicate union types. For example, `str | None` instead of
      `Optional[str]`.

## 8. Code Organization

- **Logical Separation**:
    - Group related classes and functions into modules.
    - Separate concerns — each module should handle one “purpose” or “feature” to keep the codebase maintainable.
- **Imports**:
    - Use explicit imports (avoid `from module import *`).
    - Order imports: standard library, third-party, local modules (separated by blank lines).

## 9. Linters and Code Formatters

- **Adopt a Linter** (`make lint`) to catch style issues and errors early.
- **Auto-Formatting** (`make format`) to maintain a consistent style across the codebase.

## 10. Logging and Error Handling

- **Use Standard Logging**:
    - Leverage Python’s built-in logging library instead of `print()` for production systems.
    - Configure log levels (`INFO`, `DEBUG`, `ERROR`) to separate normal behavior from error conditions.
- **Graceful Error Handling**:
    - Use exceptions wisely.
    - Avoid silent failures — log or propagate exceptions with helpful messages.


# 11. Use Pydantic models for validation and complex arguments

- ### Validation
  Use Pydantic models for data validation instead of custom validation functions. Example:
  ```python
  def validate_dict(data: dict[str, Any]) -> None:
      if 'items' not in data:
          raise ValueError("Must contain 'items' key.")

      for item in data['items']:
          if 'name' not in item or 'values' not in item:
              raise ValueError("Each dictionary must have 'name' and 'values' keys.")
  ```
  Instead, use Pydantic models:
  ```python
  from pydantic import BaseModel, StrictStr

  class Item(BaseModel):
      name: StrictStr
      values: list[StrictStr]


  class MyModel(BaseModel):
      items: list[Item]
  ```
  Then, you can validate the data using the model:
  ```python
  MyModel.model_validate(data)
  ```

- ### Complex arguments
  Use Pydantic models for complex function arguments. Example:
  ```python
  def some_function(data: dict[str, list[str, Any]]): ...

  def some_function2(items: list[dict[str, Any]]): ...
  ```
  Instead, use Pydantic models:
  ```python
  def some_function(data: MyModel): ...

  def some_function2(items: list[Item]): ...
  ```
  NOTE: It is okay to use `dict` as a mapping when the keys and values are homogeneous.
  For example, a mapping between an `id` of type `int` and the corresponding `item` of type `Item(BaseModel)`.

- ### Mutable default values in Pydantic models
  As discussed, `pydantic` can handle
  [mutable defaults values](https://docs.pydantic.dev/latest/concepts/fields/#mutable-default-values),
  but we agreed to avoid passing mutable objects as the default value of a pydantic field.
  We should use `default_factory` instead.

  Example:
  ```python
  class Model(BaseModel):
    items: list[str] = []
  ```
  Instead, use `default_factory`:
  ```python
  class Model(BaseModel):
    items: list[str] = Field(default_factory=list)
  ```


# 12. Dependencies (libraries, packages, etc.)

- **Use `poetry` for Dependency Management**:
  - Use `poetry` to manage dependencies and their versions.
  - This ensures a consistent environment across different setups and simplifies dependency resolution.
- **Direct Dependencies must be declared in `pyproject.toml`**:
  - All dependencies that are directly imported in the code must be declared in the `pyproject.toml` file.


---
