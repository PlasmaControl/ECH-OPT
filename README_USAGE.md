# ECHO Offline Optimizer - Usage Guide

## Command-Line Usage

The compiled `echo_offline_debug` executable accepts 6 command-line arguments:

```bash
./echo_offline_debug [nPop] [generations] [mutationRate] [elite] [inertia] [nCycles]
```

### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `nPop` | integer | 10-10000 | 100 | Population size for genetic algorithm |
| `generations` | integer | 1-1000 | 10 | Number of generations to evolve |
| `mutationRate` | float | 0.0-1.0 | 0.1 | Probability of mutation per gene |
| `elite` | float | 0.0-1.0 | 0.3 | Proportion of elite individuals preserved |
| `inertia` | float | 0.0-1.0 | 0.3 | Proportion of population replaced each cycle |
| `nCycles` | integer | 1-100 | 5 | Number of PCS optimization cycles |

### Examples

```bash
# Default parameters
./echo_offline_debug

# Small quick run (development)
./echo_offline_debug 50 5 0.1 0.3 0.3 2

# Larger optimization (production)
./echo_offline_debug 200 50 0.15 0.25 0.4 10

# Get help
./echo_offline_debug --help
```

## Jupyter Notebook Usage

The `plot_echo_results.ipynb` notebook has been configured to:

1. **Automatically compile** the C code on the first cell
2. **Run the optimizer** with customizable parameters in the second cell
3. **Plot and analyze** results in subsequent cells

### Workflow

1. Open `plot_echo_results.ipynb` in Jupyter
2. Run the first cell to compile `echo_offline_debug.c`
3. Modify the parameter values in the second cell:
   ```python
   nPop = 100            # Adjust population size
   generations = 10      # Adjust iterations
   mutation_rate = 0.1   # Adjust mutation rate
   elite_prop = 0.3      # Adjust elite preservation
   inertia = 0.3         # Adjust population turnover
   n_cycles = 5          # Adjust number of cycles
   ```
4. Run the second cell to execute the optimizer
5. Run remaining cells to generate plots and analysis

## Output Files

After execution, the following CSV files are generated:

- **`profile_output.csv`** - Final optimized profile vs target
  - Columns: rho, target, achieved, gyro1, gyro2, ... gyroN
  - Shows power distribution from each gyrotron

- **`profile_output_all_cycles.csv`** - All intermediate results
  - Columns: cycle, rho, target, achieved, gyro1, gyro2, ... gyroN
  - Allows tracking optimization progress

- **`gyro_parameters.csv`** - Final angles and duty cycles
  - Columns: Gyrotron, Angle, DutyCycle, PowerSupply
  - Ready for experimental implementation

## Compilation

To manually recompile:

```bash
gcc -o echo_offline_debug echo_offline_debug.c \
    echo/echo_torbeamNN_x.c echo/echo_torbeamNN_o.c \
    k2c/*.c -lm
```

## Performance Tips

- **Quick testing**: Use small `nPop` (50) and `generations` (5)
- **Publication-ready**: Use large `nPop` (200+) and `generations` (50+)
- **Multi-parameter study**: Run notebook multiple times with different parameters
- **Parallel runs**: Execute multiple notebook instances with different configs

## Troubleshooting

### "Command not found: ./echo_offline_debug"
- Ensure you're in the `/Users/ar0535/Downloads/echo_offline` directory
- Recompile with: `gcc -o echo_offline_debug echo_offline_debug.c echo/echo_torbeamNN_x.c echo/echo_torbeamNN_o.c k2c/*.c -lm`

### CSV files not created
- Check file permissions in current directory
- Ensure `profile_output.csv` in root directory (not subdirectory)

### Segmentation fault
- Memory issue likely from very large population sizes
- Reduce `nPop` and rerun

