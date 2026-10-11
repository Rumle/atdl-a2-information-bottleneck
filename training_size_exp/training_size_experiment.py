from dataclasses import replace
from shared import Settings 
from train import train_runs
# train 
def main():
    fractions = [0.03,0.1,.2,0.5,0.6,0.85]

    base_settings = Settings(
        epochs=8000,
        data_path="spherical_binary_dataset.mat",
    )

    for fraction in fractions:
        settings = replace(
            base_settings,
            train_fraction=fraction,
        )

        print(f"\nTraining with {fraction:.0%} of the data")

        train_runs(
            settings=settings,
            num_runs=50,
            run_dir=f"results/fraction-{fraction:.2f}",
            accelerator="auto",
        )


if __name__ == "__main__":
    main()