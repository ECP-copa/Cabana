# Contributing

Contributing to Cabana is easy: just open a [pull
request](https://help.github.com/articles/using-pull-requests/). Make
`master` the destination branch on the [Cabana
repository](https://github.com/ECP-copa/Cabana) and allow edits from
maintainers in the pull request.

Your pull request must pass Cabana's tests, which includes using the coding
style from `.clang-format` (enforced with clang-format-14) and adding doxygen
documentation, and be reviewed by at least one Cabana developer.

`pre-commit` is a useful tool for ensuring feature branches are ready for
review by running automatic checks locally before a commit is made.
[Installation details](https://pre-commit.com/#install) (once per system) and
[activation details](https://pre-commit.com/#usage) (once per repo) are
available.

Other coding style includes:
* Camel case template parameters (`NewTemplateType`)
* Camel case class names (`NewClassName`)
* Lower camel case function names (`newFunctionName`)
  * Note: there are some exceptions to match Kokkos (e.g. `Cabana::deep_copy`
    and `Cabana::neighbor_parallel_for`)
* Lower case, underscore separated variables (`new_variable_name`)
* Class members which are `private` are preceded by an underscore (`_private_class_variable`)
* Class/struct member type aliases use lower case, underscore separated names (`using integer_type = int;`)
