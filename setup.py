from ard_srctool.package import setup_codebase

setup_codebase(
    name="ard-fit-pipelines",
    description="A new codebase.",
    package_dir={"": "src"},
    packages=["ard_fit_pipelines"],
    install_requires_upstream=["pandas"],
    install_requires_codebases=["ard-logging"],
)
