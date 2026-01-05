/*
 * ECHO Offline Debug - Direct extraction from echo_master.h
 * This mirrors the actual runtime code as closely as possible
 * Compile: gcc -o echo_offline_debug echo_offline_debug.c echo/echo_torbeamNN_x.c echo/echo_torbeamNN_o.c k2c/k2c_*.c -lm && ./echo_offline_debug
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// K2C includes for TorbeamNN
#include "k2c/k2c_tensor_include.h"
#include "echo/echo_torbeamNN_x.h"
#include "echo/echo_torbeamNN_o.h"

// Constants
#define ECHO_NGYROS 11
#define ECHO_NGYROPS 5
#define ECHO_PROFILESIZE 101
#define NUM_ECHO_PCA_ROWS 101
#define NUM_ECHO_PCA_COLS 4
#define NUM_ECHO_NORM_ROWS 24
#define NUM_ECHO_DENORM_ROWS 3
#define ECHO_STEPMAX 100
#define ECHO_GENMAX 1048576
#define ECHO_POPMAX 1000
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif
#define FLT_MAX 3.402823466e+38F
#define NUM_ECHO_DENORM 2
#define NUM_ECHO_NORM 2

enum Echo_ProfileType {
    EchoProfileECH = 0,
    EchoProfileECCD = 1
};

// Structures from echo_master.h
struct echo_gyroen {
    int gyroEnable[ECHO_NGYROS];
};

struct echo_modetype {
    int mode;
};

struct echo_debugTor {
    float angle[ECHO_NGYROS];
};

struct echo_params {
    int size;
    int ngen;
    int tournament;
    float mutationRate;
    float elite;
    float inertia;
};

struct echo_scratch {
    size_t randomIndex;
    long randomTable[ECHO_GENMAX];
    
    size_t nGyros;
    size_t gyroGIndexes[ECHO_NGYROS];
    float gyroPower[ECHO_NGYROS];
    int gyroPs[ECHO_NGYROS];
    
    void *pool;
    size_t nWorkers;
    float torbeamIn[NUM_ECHO_NORM_ROWS];
    float nnResults[ECHO_NGYROS][ECHO_STEPMAX][ECHO_PROFILESIZE * 2];
    
    size_t angStepsMax[ECHO_NGYROS];
    float angStepNorm[ECHO_NGYROS];
    float angMinNorm[ECHO_NGYROS];
    float angMaxNorm[ECHO_NGYROS];
    size_t angStepWork[ECHO_NGYROS];
    float angPol[ECHO_NGYROS][ECHO_STEPMAX];
    float angTor[ECHO_NGYROS];
};

struct echo_stateprofiles {
    float best[ECHO_NGYROS][NUM_ECHO_PCA_ROWS];
    float target[NUM_ECHO_PCA_ROWS];
};

struct Individual {
    size_t step[ECHO_NGYROS];
    float angles[ECHO_NGYROS];
    float dutyCycles[ECHO_NGYROPS];
    float fitness;
};

typedef float echo_aranges[ECHO_NGYROS][3];
typedef float echo_order[ECHO_NGYROS][1];
typedef float echo_pslimits[ECHO_NGYROPS][2];
typedef float echo_norm[NUM_ECHO_NORM_ROWS][2];
typedef float echo_denorm[NUM_ECHO_DENORM][NUM_ECHO_DENORM_ROWS][2];

// Helper functions
static inline float kToolMath_distributionGauss(float amp, float mean, float std, float x) {
    return amp * expf(-0.5f * powf((x - mean) / std, 2.0f));}

static inline float kToolMath_normalize(float x, float mean, float std) {
    return (x - mean) / std;
}

static inline float kToolMath_denormalize(float mean, float std, float x) {
    return x * std + mean;
}

static inline float kToolRt_clamp(float min, float max, float val) {
    float t = val < min ? min : val;
    return t > max ? max : t;
}

static void kToolRt_randomInit(unsigned int seed, size_t n, long *table) {
    srandom(seed);
    for (size_t i = 0; i < n; ++i)
        table[i] = random();
}

static long kToolRt_random(size_t *index, size_t n, long const *table) {
    *index = (*index + 1) % n;
    return table[*index];
}

static long kToolRt_randomRange(size_t *index, size_t n, long const *table, long min, long max) {
    return (kToolRt_random(index, n, table) % (max - min + 1)) + min;
}

static float kToolRt_randomFloat(size_t *index, size_t n, long const *table, float min, float max) {
    return ((double)kToolRt_random(index, n, table)) / RAND_MAX * (max - min) + min;
}

static size_t kToolRt_atomicFetchAdd(size_t *ptr, size_t val) {
    size_t old = *ptr;
    *ptr += val;
    return old;
}

// Global state
static bool echoFirst = true;

//============================================================================
// ECHO FUNCTIONS - EXTRACTED DIRECTLY FROM echo_master.h
//============================================================================

void echo_init(struct echo_scratch *scratch, echo_aranges const *aRanges,
               struct echo_gyroen const *gyroen, echo_order const *order,
               struct echo_debugTor const *tor, unsigned int seed) {
    // From echo_master.h lines 648-697
    kToolRt_randomInit(seed, ECHO_GENMAX, scratch->randomTable);
    
    for (size_t i = 0; i < ECHO_NGYROS; ++i) {
        enum {
            AngleMin,
            AngleMax,
            AngleStep,
        };
        
        float angle = (*aRanges)[i][AngleMin];
        float const angleMax = (*aRanges)[i][AngleMax];
        float const step = (*aRanges)[i][AngleStep];
        scratch->angStepNorm[i] = step;
        scratch->angMinNorm[i] = angle;
        scratch->angMaxNorm[i] = angleMax;
        
        for (size_t j = 0; j < ECHO_STEPMAX && angle <= angleMax; ++j) {
            scratch->angPol[i][j] = angle;
            scratch->angStepsMax[i] = j;
            angle += step;
        }
        
        scratch->angTor[i] = tor->angle[i];
    }
    
    size_t nGyros = 0;
    for (size_t i = 0; i < ECHO_NGYROS; ++i) {
        if (gyroen->gyroEnable[i])
            scratch->gyroGIndexes[nGyros++] = (*order)[i][0] - 1;
    }
    scratch->nGyros = nGyros;
    
    printf("Initialized with %zu active gyros: ", nGyros);
    for (size_t i = 0; i < nGyros; ++i) {
        printf("G%zu ", scratch->gyroGIndexes[i] + 1);
    }
    printf("\n");
    
    scratch->nWorkers = 1; // Single-threaded offline
    scratch->pool = NULL;
    
    // Initialize k2c neural network models
    echo_torbeamNN_x_initialize();
    echo_torbeamNN_o_initialize();
}

void echo_phaseenter(void) {
    echoFirst = true;
}

// Helper enums and function for normalization
enum {
    Mean,
    Std,
};

enum NormField {
    NormPol,
    NormTor,
};

static inline float normalize(enum NormField field, float x, echo_norm const *norm) {
    float const mean = (*norm)[field][Mean];
    float const std_val = (*norm)[field][Std];
    return kToolMath_normalize(x, mean, std_val);
}

//============================================================================
// OPTIMIZER HELPER FUNCTIONS
//============================================================================

static struct Individual population[ECHO_POPMAX];

static inline long int opt_randomInt(struct echo_scratch *scratch) {
    return kToolRt_random(&scratch->randomIndex, ECHO_GENMAX, scratch->randomTable);
}

static inline long int opt_randomIntRange(struct echo_scratch *scratch, long int min, long int max) {
    return kToolRt_randomRange(&scratch->randomIndex, ECHO_GENMAX, scratch->randomTable, min, max);
}

static inline float opt_randomFloat(struct echo_scratch *scratch, float min, float max) {
    return kToolRt_randomFloat(&scratch->randomIndex, ECHO_GENMAX, scratch->randomTable, min, max);
}

static void opt_setAngle(struct echo_scratch *scratch, size_t nGyros, struct Individual *ind, 
                         size_t echoGyroIndex, float lowerBound, float upperBound, float angle) {
    size_t const i = echoGyroIndex;
    size_t const gIndex = scratch->gyroGIndexes[i];
    float const step = scratch->angStepNorm[gIndex];
    float const min = scratch->angMinNorm[gIndex];
    
    ind->angles[i] = kToolRt_clamp(lowerBound, upperBound, roundf((angle - min) / step) * step + min);
    ind->step[i] = (size_t)roundf((ind->angles[i] - min) / step);
}

static void opt_setAngleRandom(struct echo_scratch *scratch, size_t nGyros, struct Individual *ind, size_t i) {
    size_t gIndex = scratch->gyroGIndexes[i];
    float lowerBound = scratch->angMinNorm[gIndex];
    if (i > 0)
        lowerBound = fmaxf(lowerBound, ind->angles[i - 1]);
    
    float upperBound = scratch->angMaxNorm[gIndex];
    if (i < nGyros - 1)
        upperBound = fminf(upperBound, ind->angles[i + 1]);
    
    opt_setAngle(scratch, nGyros, ind, i, lowerBound, upperBound, 
                 opt_randomFloat(scratch, lowerBound, upperBound));
}

static struct Individual opt_createIndividual(struct echo_scratch *scratch, size_t nGyros, 
                                               echo_pslimits const *pslimits) {
    struct Individual ind;
    
    for (size_t i = 0; i < nGyros; ++i)
        opt_setAngleRandom(scratch, nGyros, &ind, i);
    
    for (size_t i = 0; i < ECHO_NGYROPS; ++i)
        ind.dutyCycles[i] = opt_randomFloat(scratch, (*pslimits)[i][0], (*pslimits)[i][1]);
    
    return ind;
}

static void opt_setFitness(struct echo_scratch *scratch, size_t nGyros, struct Individual *ind,
                           float const profileTarget[NUM_ECHO_PCA_ROWS], enum Echo_ProfileType depProfile) {
    float fitness = 0.0f;
    for (size_t i = 0; i < NUM_ECHO_PCA_ROWS; ++i) {
        float fitnessProfile = 0.0f;
        for (size_t j = 0; j < nGyros; ++j) {
            size_t const gIndex = scratch->gyroGIndexes[j];
            float const scale = 1.0e-6f * scratch->gyroPower[gIndex] * ind->dutyCycles[scratch->gyroPs[gIndex]];
            fitnessProfile += scale * scratch->nnResults[gIndex][ind->step[j]][i + NUM_ECHO_PCA_ROWS * depProfile];
        }
        
        float const delta = fitnessProfile - profileTarget[i];
        fitness += delta * delta;
    }
    
    ind->fitness = fitness;
}

static void opt_crossover(struct echo_scratch *scratch, size_t nGyros,
                         struct Individual const *parent1, struct Individual const *parent2,
                         struct Individual *child1, struct Individual *child2) {
    struct Individual const *parents[] = {parent1, parent2};
    struct Individual *children[] = {child1, child2};
    
    for (size_t i = 0; i < nGyros; ++i) {
        size_t const gIndex = scratch->gyroGIndexes[i];
        float const min = scratch->angMinNorm[gIndex];
        float const max = scratch->angMaxNorm[gIndex];
        float const alpha = opt_randomFloat(scratch, 0.0f, 1.0f);
        for (size_t j = 0; j < 2; ++j) {
            float const angle = alpha * parents[j]->angles[i] + (1.0f - alpha) * parents[1 - j]->angles[i];
            opt_setAngle(scratch, nGyros, children[j], i, min, max, angle);
        }
    }
    
    for (size_t i = 0; i < ECHO_NGYROPS; ++i) {
        float const alpha = opt_randomFloat(scratch, 0.0f, 1.0f);
        for (size_t j = 0; j < 2; ++j)
            children[j]->dutyCycles[i] = alpha * parents[j]->dutyCycles[i] + (1.0f - alpha) * parents[1 - j]->dutyCycles[i];
    }
}

static void opt_mutate(struct echo_scratch *scratch, size_t nGyros, struct echo_params const *params,
                      echo_pslimits const *pslimits, struct Individual *ind) {
    for (size_t i = 0; i < nGyros; ++i)
        if (opt_randomFloat(scratch, 0.0f, 1.0f) < params->mutationRate)
            opt_setAngleRandom(scratch, nGyros, ind, i);
    
    for (size_t i = 0; i < ECHO_NGYROPS; ++i)
        if (opt_randomFloat(scratch, 0.0f, 1.0f) < params->mutationRate)
            ind->dutyCycles[i] = opt_randomFloat(scratch, (*pslimits)[i][0], (*pslimits)[i][1]);
}

static int opt_comparePopDesc(void const *A, void const *B) {
    struct Individual const *a = A;
    struct Individual const *b = B;
    return (b->fitness > a->fitness) - (b->fitness < a->fitness);
}

static void opt_sortPop(size_t nPop) {
    qsort(population, nPop, sizeof(*population), opt_comparePopDesc);
}

static void opt_updateFitness(struct echo_scratch *scratch, size_t nGyros, size_t nPop,
                              float const profileTarget[NUM_ECHO_PCA_ROWS], enum Echo_ProfileType depProfile) {
    for (size_t i = 0; i < nPop; ++i)
        opt_setFitness(scratch, nGyros, &population[i], profileTarget, depProfile);
}

void echo_torbeam(struct echo_scratch *scratch, struct echo_gyroen const *gyroen,
                  bool use_torbeamNN, int mode, echo_norm const *norm, echo_denorm const *denorm) {
    // From echo_master.h lines 867-932 (echo_worker function)
    
    for (size_t i = 0; i < ECHO_NGYROS; ++i) {
        if (!gyroen->gyroEnable[i])
            continue;
        
        size_t j = kToolRt_atomicFetchAdd(&scratch->angStepWork[i], 1);
        if (j < scratch->angStepsMax[i]) {
            float const angle = scratch->angPol[i][j];
            
            if (!use_torbeamNN) {
                // Gaussian simulation (debug mode)
                float const center = 0.8f + 0.04f * (100.0f - angle);
                float const std = 1.0f / (angle - 92.0f);
                float const area = 1.2533f * std * (erff(center / std * M_SQRT1_2) - erff((center - 1.0f) / std * M_SQRT1_2));
                
                for (size_t k = 0; k < NUM_ECHO_PCA_ROWS; ++k) {
                    float const x = k / (float)(NUM_ECHO_PCA_ROWS - 1);
                    float const gaussian = 0.5f / area * expf(-0.5f * powf((x - center) / std, 2.0f));
                    scratch->nnResults[i][j][k + 0] = gaussian;                 // ECH
                    scratch->nnResults[i][j][k + NUM_ECHO_PCA_ROWS] = gaussian; // ECCD
                }
            } else {
                // TorbeamNN mode
                float torbeamIn[NUM_ECHO_NORM_ROWS];
                
                // Copy and normalize ALL parameters (not just angles)
                for (size_t k = 0; k < NUM_ECHO_NORM_ROWS; ++k) {
                    torbeamIn[k] = kToolMath_normalize(scratch->torbeamIn[k], (*norm)[k][Mean], (*norm)[k][Std]);
                }
                
                // Angles: use scratch->angTor and current angle, then normalize
                float const rawPol = angle;
                float const rawTor = scratch->angTor[i];
                
                // Rotate, convert to radians, THEN normalize
                torbeamIn[NormPol] = normalize(NormPol, M_PI / 180.0f * (rawPol - 90.0f), norm);
                torbeamIn[NormTor] = normalize(NormTor, M_PI / 180.0f * (180.0f - rawTor), norm);
                
                // Debug: Print first call
                if (i == 3 && j == 0) {
                    printf("DEBUG TorbeamNN input for G%zu, step 0:\n", i+1);
                    printf("  Raw: pol=%.2f deg, tor=%.2f deg\n", rawPol, rawTor);
                    printf("  Normalized pol=%.6f, tor=%.6f\n", torbeamIn[NormPol], torbeamIn[NormTor]);
                    printf("  torbeamIn[2-5]=%.4f,%.4f,%.4f,%.4f\n", 
                           torbeamIn[2], torbeamIn[3], torbeamIn[4], torbeamIn[5]);
                }
                
                // Build k2c tensors
                float results[NUM_ECHO_DENORM][NUM_ECHO_DENORM_ROWS];
                struct k2c_tensor in = {torbeamIn, 1, NUM_ECHO_NORM_ROWS, {NUM_ECHO_NORM_ROWS}};
                struct k2c_tensor out = {results[0], 1, NUM_ECHO_DENORM * NUM_ECHO_DENORM_ROWS, {NUM_ECHO_DENORM * NUM_ECHO_DENORM_ROWS}};
                
                // Call appropriate model
                enum { ModeX = 0, ModeO = 1 };
                switch (mode) {
                    case ModeX: echo_torbeamNN_x(&in, &out); break;
                    case ModeO: echo_torbeamNN_o(&in, &out); break;
                    default: break;
                }
                
                // Debug: Print output
                if (i == 3 && j == 0) {
                    printf("  TorbeamNN output (raw): ECH[mean=%.6f, amp=%.6f, std=%.6f]\n",
                           results[0][0], results[0][1], results[0][2]);
                }
                
                // Denormalize and apply Gaussian distribution
                for (size_t profile = 0; profile < NUM_ECHO_DENORM; ++profile) {
                    for (size_t k = 0; k < NUM_ECHO_DENORM_ROWS; ++k)
                        results[profile][k] = kToolMath_denormalize((*denorm)[profile][k][Mean], (*denorm)[profile][k][Std], results[profile][k]);
                    
                    if (i == 3 && j == 0 && profile == 0) {
                        printf("  After denorm: ECH[mean=%.6f, amp=%.6f, std=%.6f]\n",
                               results[0][0], results[0][1], results[0][2]);
                    }
                    
                    for (size_t k = 0; k < NUM_ECHO_PCA_ROWS; ++k) {
                        enum {
                            OutMean,
                            OutAmp,
                            OutStd,
                        };
                        float const amp = results[profile][OutAmp];
                        float const mean = results[profile][OutMean];
                        float const std = results[profile][OutStd];
                        float const x = k / (float)(NUM_ECHO_PCA_ROWS - 1);
                        scratch->nnResults[i][j][k + profile * NUM_ECHO_PCA_ROWS] = kToolMath_distributionGauss(amp, mean, std, x);
                    }
                }
            }
        }
    }
}

void echo_optimizer(struct echo_scratch *scratch,
                    struct echo_gyroen const *gyroen,
                    echo_order const *order,
                    struct echo_params const *params,
                    echo_pslimits const *pslimits,
                    float const profileTarget[NUM_ECHO_PCA_ROWS],
                    enum Echo_ProfileType depProfile,
                    float *outAngles,
                    float *outDutyCycles,
                    float *outFitness,
                    struct echo_stateprofiles *profiles) {
    
    // Extracted from echo_master.h lines 977-1213
    
    size_t const nGyros = scratch->nGyros;
    size_t const nPop = params->size;
    
    // Initialize population
    if (echoFirst) {
        echoFirst = false;
        for (size_t i = 0; i < nPop; ++i)
            population[i] = opt_createIndividual(scratch, nGyros, pslimits);
        opt_updateFitness(scratch, nGyros, nPop, profileTarget, depProfile);
        opt_sortPop(nPop);
    }
    
    float const inertia = params->inertia;
    for (size_t i = 0; i < nPop * (1.0f - inertia); ++i) {
        population[i] = opt_createIndividual(scratch, nGyros, pslimits);
        opt_setFitness(scratch, nGyros, &population[i], profileTarget, depProfile);
    }
    
    static struct Individual populationNext[ECHO_POPMAX];
    for (size_t generation = 0; generation < params->ngen; ++generation) {
        opt_sortPop(nPop);
        size_t const elite = (1.0f - params->elite) * nPop;
        
        for (size_t i = 0; i < elite; i += 2) {
            size_t parent1Index = opt_randomIntRange(scratch, 0, nPop - 1);
            size_t parent2Index = opt_randomIntRange(scratch, 0, nPop - 1);
            
            for (size_t j = 0; j < params->tournament; ++j) {
                size_t index = opt_randomIntRange(scratch, 0, nPop - 1);
                float const fitness = population[index].fitness;
                if (fitness < population[parent1Index].fitness) {
                    parent2Index = parent1Index;
                    parent1Index = index;
                } else if (fitness < population[parent2Index].fitness && fitness != population[parent1Index].fitness) {
                    parent2Index = index;
                }
            }
            
            struct Individual child1, child2;
            opt_crossover(scratch, nGyros, &population[parent1Index], &population[parent2Index], &child1, &child2);
            opt_mutate(scratch, nGyros, params, pslimits, &child1);
            opt_mutate(scratch, nGyros, params, pslimits, &child2);
            
            populationNext[i] = child1;
            if (i + 1 < elite)
                populationNext[i + 1] = child2;
        }
        
        for (size_t i = elite; i < nPop; ++i)
            populationNext[i] = population[i];
        
        for (size_t i = 0; i < nPop; ++i)
            population[i] = populationNext[i];
        opt_updateFitness(scratch, nGyros, nPop, profileTarget, depProfile);
    }
    
    opt_sortPop(nPop);
    struct Individual const *const best = &population[nPop - 1];
    
    // Output results
    for (size_t i = 0; i < ECHO_NGYROS; ++i)
        outAngles[i] = 0.0f;
    for (size_t i = 0; i < nGyros; ++i)
        outAngles[scratch->gyroGIndexes[i]] = best->angles[i];
    
    for (size_t i = 0; i < ECHO_NGYROPS; ++i)
        outDutyCycles[i] = best->dutyCycles[i];
    
    *outFitness = best->fitness;
    
    // Store profiles
    for (size_t i = 0; i < NUM_ECHO_PCA_ROWS; ++i)
        profiles->target[i] = profileTarget[i];
    
    for (size_t i = 0; i < nGyros; ++i) {
        size_t const gIndex = scratch->gyroGIndexes[i];
        float const scale = 1.0e-6f * scratch->gyroPower[gIndex] * best->dutyCycles[scratch->gyroPs[gIndex]];
        for (size_t j = 0; j < NUM_ECHO_PCA_ROWS; ++j)
            profiles->best[gIndex][j] = scale * scratch->nnResults[gIndex][best->step[i]][j + NUM_ECHO_PCA_ROWS * depProfile];
    }
    
    // Zero out disabled gyros
    for (size_t i = 0; i < ECHO_NGYROS; ++i) {
        if (!gyroen->gyroEnable[i]) {
            for (size_t j = 0; j < NUM_ECHO_PCA_ROWS; ++j)
                profiles->best[i][j] = 0.0f;
        }
    }
}

//============================================================================
// MAIN - TEST HARNESS
//============================================================================

int main(int argc, char *argv[]) {
    // Default parameter values
    int nPop = 100;
    int nGen = 10;
    float mutationRate = 0.1f;
    float elite = 0.3f;
    float inertia = 0.3f;
    int nCycles = 5;
    char profileFile[256] = "profile_target.txt";
    float angmin = 100.0f;
    float angmax = 120.0f;
    float angstep = 0.5f;
    float dc_min = 0.0f;
    float dc_max = 1.0f;
    float bt_scale = 1.0f;
    int mode = 0; // 0=X-Mode, 1=O-Mode
    
    // Parse command-line arguments
    if (argc > 1) nPop = atoi(argv[1]);
    if (argc > 2) nGen = atoi(argv[2]);
    if (argc > 3) mutationRate = atof(argv[3]);
    if (argc > 4) elite = atof(argv[4]);
    if (argc > 5) inertia = atof(argv[5]);
    if (argc > 6) nCycles = atoi(argv[6]);
    if (argc > 7) strncpy(profileFile, argv[7], sizeof(profileFile) - 1);
    if (argc > 8) angmin = atof(argv[8]);
    if (argc > 9) angmax = atof(argv[9]);
    if (argc > 10) angstep = atof(argv[10]);
    if (argc > 11) dc_min = atof(argv[11]);
    if (argc > 12) dc_max = atof(argv[12]);
    if (argc > 13) bt_scale = atof(argv[13]);
    if (argc > 14) mode = atoi(argv[14]);
    
    // Print usage if requested
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s [nPop] [generations] [mutationRate] [elite] [inertia] [nCycles] [profileFile] [angmin] [angmax] [angstep] [dc_min] [dc_max] [bt_scale] [mode]\n", argv[0]);
        printf("  nPop:          Population size (default: 100)\n");
        printf("  generations:   Number of generations (default: 10)\n");
        printf("  mutationRate:  Mutation rate 0-1 (default: 0.1)\n");
        printf("  elite:         Elite proportion 0-1 (default: 0.3)\n");
        printf("  inertia:       Inertia 0-1 (default: 0.3)\n");
        printf("  nCycles:       Number of PCS cycles (default: 5)\n");
        printf("  profileFile:   Target profile text file with 101 float values (default: profile_target.txt)\n");
        printf("  angmin:        Minimum gyrotron angle in degrees (default: 100.0)\n");
        printf("  angmax:        Maximum gyrotron angle in degrees (default: 120.0)\n");
        printf("  angstep:       Angle step size in degrees (default: 0.5)\n");
        printf("  dc_min:        Minimum duty cycle 0-1 (default: 0.0)\n");
        printf("  dc_max:        Maximum duty cycle 0-1 (default: 1.0)\n");
        printf("  bt_scale:      Toroidal field scaling factor (default: 1.0)\n");
        printf("  mode:          Polarization mode: 0=X-Mode, 1=O-Mode (default: 0)\n");
        printf("\nExample: %s 50 20 0.15 0.25 0.4 3 my_profile.txt 100 120 0.5 0.0 1.0 1.0 0\n", argv[0]);
        return 0;
    }
    
    printf("================================================\n");
    printf("ECHO Debug Mode (Extracted from echo_master.h)\n");
    printf("================================================\n\n");
    
    // ========================================
    // CONFIGURATION - Set use_torbeamNN here!
    // ========================================
    bool use_torbeamNN = true;  // Set to true to use TorbeamNN, false for Gaussian
    
    printf("Mode: %s\n", use_torbeamNN ? "TorbeamNN" : "Gaussian");
    if (use_torbeamNN)
        printf("Polarization: %s\n", mode == 0 ? "X-Mode" : "O-Mode");
    printf("Parameters: pop=%d, gen=%d, mutation=%.3f, elite=%.3f, inertia=%.3f, cycles=%d\n",
           nPop, nGen, mutationRate, elite, inertia, nCycles);
    printf("Angle range: %.1f° to %.1f° (step: %.2f°)\n", angmin, angmax, angstep);
    printf("Duty cycle limits: [%.2f, %.2f]\n", dc_min, dc_max);
    printf("Toroidal field scale: %.2f\n", bt_scale);
    printf("\n");
    
    // Configuration
    echo_aranges aRanges = {
        {angmin, angmax, angstep}, {angmin, angmax, angstep}, {angmin, angmax, angstep},
        {angmin, angmax, angstep}, {angmin, angmax, angstep}, {angmin, angmax, angstep},
        {angmin, angmax, angstep}, {angmin, angmax, angstep}, {angmin, angmax, angstep},
        {angmin, angmax, angstep}, {angmin, angmax, angstep}
    };
    
    struct echo_gyroen gyroen = {
        .gyroEnable = {0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1}
    };
    
    echo_order order = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}
    };
    
    struct echo_debugTor tor = {
        .angle = {180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180}
    };
    
    struct echo_params params = {
        .size = nPop,       
        .ngen = nGen,
        .tournament = 10,
        .mutationRate = mutationRate,
        .elite = elite,
        .inertia = inertia
    };
    
    echo_pslimits pslimits = {
        // {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}
        {dc_min, dc_max}, {dc_min, dc_max}, {dc_min, dc_max}, {dc_min, dc_max}, {dc_min, dc_max},
    };
    
    // Normalization data (from input_normalization.pkl)
    echo_norm norm = {
        {0.343084f, 0.239022f}, // pol_ang
        {-0.000415f, 0.291869f}, // tor_ang
        {0.590272f, 0.019816f}, // AMINOR
        {0.173664f, 0.089394f}, // GAPBOT
        {0.07116f, 0.036909f}, // GAPIN
        {0.081218f, 0.019585f}, // GAPOUT
        {0.210446f, 0.085461f}, // GAPTOP
        {1.776675f, 0.106819f}, // KAPPA
        {0.983908f, 0.245382f}, // LI
        {0.538916f, 0.191562f}, // TRIBOT
        {0.458269f, 0.190284f}, // TRITOP
        {18.288182f, 1.116639f}, // VOLUME
        {1.755439f, 0.03711f}, // R0
        {-0.019146f, 0.036248f}, // Z0
        {1016314.524711f, 250863.04884f}, // IPMHD
        {111329.435649f, 9866.595482f}, // PCBCOIL
        {-9.186037f, 2.845769f}, // Te_0
        {-14.936721f, 4.915148f}, // Te_1
        {-1.614087f, 0.927423f}, // Te_2
        {0.332675f, 0.490201f}, // Te_3
        {27.264121f, 8.751962f}, // ne_0
        {31.569648f, 10.576687f}, // ne_1
        {-0.297149f, 2.250038f}, // ne_2
        {1.784233f, 1.183503f}, // ne_3
    };
    
    // Denormalization data for outputs (from output_normalization.pkl)
    echo_denorm denorm = {
        {  // ECH profile (index 0)
            {0.551518f, 0.208641f},   // power_dens_mu (mean, std)
            {0.652025f, 0.362307f},   // power_dens_amp (mean, std)
            {0.035008f, 0.015139f}    // power_dens_sigma (mean, std)
        },
        {  // ECCD profile (index 1)
            {0.551518f, 0.208641f},   // curr_dens_mu (mean, std)
            {0.652025f, 0.362307f},   // curr_dens_amp (mean, std)
            {0.035008f, 0.015139f}    // curr_dens_sigma (mean, std)
        }
    };
    
    // Allocate structures (static to avoid stack overflow)
    static struct echo_scratch scratch = {0};
    static struct echo_stateprofiles profiles = {0};
    
    // Set gyrotron powers and PS assignments
    const int gyroPs[ECHO_NGYROS] = {
        0, // G1  -> PS1
        0, // G2  -> PS1
        0, // G3  -> PS1
        1, // G4  -> PS2
        2, // G5  -> PS3
        0, // G6  -> PS1
        0, // G7  -> PS1
        2, // G8  -> PS3
        4, // G9  -> PS5
        0, // G10 -> PS1
        3  // G11 -> PS4
    };
    
    const float gyroPower[ECHO_NGYROS] = {
        0.0f,      // G1
        0.0f,      // G2
        0.0f,      // G3
        500000.0f, // G4
        600000.0f, // G5
        0.0f,      // G6
        0.0f,      // G7
        550000.0f, // G8
        450000.0f, // G9
        0.0f,      // G10
        400000.0f  // G11
    };
    
    for (int i = 0; i < ECHO_NGYROS; ++i) {
        scratch.gyroPower[i] = gyroPower[i];
        scratch.gyroPs[i] = gyroPs[i];
    }
    
    // Initialize plasma parameters for TorbeamNN (if used)
    if (use_torbeamNN) {
        // Real plasma parameters from shot data
        // These are the raw (unnormalized) values that will be normalized in echo_torbeam
        enum {
            Pol, Tor, AMinor, GapBot, GapIn, GapOut, GapTop, Kappa, Li, TriBot, TriTop,
            Vol, R0, Z0, IpMhd, BCoil, Te1, Te2, Te3, Te4, Ne1, Ne2, Ne3, Ne4
        };
        
        // Angles (will be set per angle step and gyro)
        scratch.torbeamIn[Pol] = 0.0f;           // [0] Poloidal: 100-125 deg (set per angle step)
        scratch.torbeamIn[Tor] = 0.0f;           // [1] Toroidal: 200 deg (set per gyro)
        
        // Equilibrium parameters
        scratch.torbeamIn[AMinor] = 0.5657f;     // [2] Minor radius (m)
        scratch.torbeamIn[GapBot] = 0.1263f;     // [3] Bottom gap (m)
        scratch.torbeamIn[GapIn] = 0.0835f;      // [4] Inner gap (m)
        scratch.torbeamIn[GapOut] = 0.08548f;    // [5] Outer gap (m)
        scratch.torbeamIn[GapTop] = 0.426f;      // [6] Top gap (m)
        scratch.torbeamIn[Kappa] = 1.837f;       // [7] Elongation
        scratch.torbeamIn[Li] = 0.917f;          // [8] Internal inductance
        scratch.torbeamIn[TriBot] = 0.768f;      // [9] Bottom triangularity
        scratch.torbeamIn[TriTop] = 0.203f;      // [10] Top triangularity
        scratch.torbeamIn[Vol] = 17.519f;        // [11] Volume (m^3)
        scratch.torbeamIn[R0] = 1.804f;          // [12] Major radius (m)
        scratch.torbeamIn[Z0] = 0.0114f;         // [13] Vertical position (m)
        
        // Current and field
        scratch.torbeamIn[IpMhd] = 941439.06f;   // [14] Plasma current (A)
        scratch.torbeamIn[BCoil] = 111488.0f * bt_scale;    // [15] Toroidal field coil current (A) - scaled
        
        // PCA components for Te profile
        scratch.torbeamIn[Te1] = -9.186f;        // [16] Te PCA component 1
        scratch.torbeamIn[Te2] = -14.937f;       // [17] Te PCA component 2
        scratch.torbeamIn[Te3] = -1.614f;        // [18] Te PCA component 3
        scratch.torbeamIn[Te4] = 0.3326f;        // [19] Te PCA component 4
        
        // PCA components for ne profile
        scratch.torbeamIn[Ne1] = 27.26f;         // [20] ne PCA component 1
        scratch.torbeamIn[Ne2] = 31.57f;         // [21] ne PCA component 2
        scratch.torbeamIn[Ne3] = -0.297f;        // [22] ne PCA component 3
        scratch.torbeamIn[Ne4] = 1.784f;         // [23] ne PCA component 4
    }
    
    // Target profile - read from file
    static float profileTarget[NUM_ECHO_PCA_ROWS];
    FILE *fp_target = fopen(profileFile, "r");
    if (!fp_target) {
        fprintf(stderr, "Error: Cannot open target profile file '%s'\n", profileFile);
        fprintf(stderr, "Please provide a text file with %d float values (one per line)\n", NUM_ECHO_PCA_ROWS);
        return 1;
    }
    
    for (int i = 0; i < NUM_ECHO_PCA_ROWS; ++i) {
        if (fscanf(fp_target, "%f", &profileTarget[i]) != 1) {
            fprintf(stderr, "Error: Could not read value %d from '%s'\n", i + 1, profileFile);
            fprintf(stderr, "Expected %d float values\n", NUM_ECHO_PCA_ROWS);
            fclose(fp_target);
            return 1;
        }
    }
    fclose(fp_target);
    printf("✓ Loaded target profile from '%s'\n\n", profileFile);
    
    // Initialize
    printf("Initializing...\n");
    echo_init(&scratch, &aRanges, &gyroen, &order, &tor, 10101);
    echo_phaseenter();
    
    // Run torbeam - need to call multiple times for each angle step
    int maxSteps = 0;
    for (int i = 0; i < ECHO_NGYROS; ++i) {
        if (scratch.angStepsMax[i] > maxSteps)
            maxSteps = scratch.angStepsMax[i];
    }
    printf("Running torbeam calculation (%s) for %d angle steps...\n", 
           use_torbeamNN ? "TorbeamNN" : "Gaussian", maxSteps + 1);
    for (int step = 0; step <= maxSteps; ++step) {
        echo_torbeam(&scratch, &gyroen, use_torbeamNN, mode, &norm, &denorm);
    }
    
    // Debug: Check that all steps were computed
    printf("DEBUG: Checking that all angle steps were computed...\n");
    for (int g = 0; g < 5; ++g) {
        int gIndex = scratch.gyroGIndexes[g];
        printf("  Gyro G%d (index %d): angStepWork=%zu, angStepsMax=%zu\n",
               gIndex+1, gIndex, scratch.angStepWork[gIndex], scratch.angStepsMax[gIndex]);
    }
    
    // Debug: Check nnResults for one gyro
    printf("DEBUG echo_offline_debug.c: Sample nnResults for G4 at angle step 10 (110 deg), rho=0.33 (index 33):\n");
    printf("  nnResults[3][10][33] (ECH) = %f\n", scratch.nnResults[3][10][33]);
    printf("  nnResults[3][10][134] (ECCD) = %f\n", scratch.nnResults[3][10][134]);
    printf("  With gyroPower=%.2f and dutyCycle=0.7, scale=%.3f, contribution=%.3f\n",
           scratch.gyroPower[3], scratch.gyroPower[3] * 0.7f, scratch.gyroPower[3] * 0.7f * scratch.nnResults[3][10][33]);
    
    printf("DEBUG: Sample nnResults for G4 (index 3) at different angles:\n");
    for (int step = 0; step <= 20; step += 5) {
        float maxval = 0.0f;
        int maxidx = 0;
        for (int i = 0; i < NUM_ECHO_PCA_ROWS; ++i) {
            if (scratch.nnResults[3][step][i] > maxval) {
                maxval = scratch.nnResults[3][step][i];
                maxidx = i;
            }
        }
        printf("  Step %d (angle %.0f deg): max=%.6f at rho=%.3f\n", 
               step, scratch.angPol[3][step], maxval, maxidx/(float)(NUM_ECHO_PCA_ROWS-1));
    }
    printf("\n");
    
    // Run optimizer for multiple PCS cycles
    printf("Running optimizer (pop=%d, gen=%d, inertia=%.2f)...\n", params.size, params.ngen, params.inertia);
    float outAngles[ECHO_NGYROS];
    float outDutyCycles[ECHO_NGYROPS];
    float outFitness;
    
    // Open CSV file for all cycles
    FILE *fp_all = fopen("csv_files/profile_output_all_cycles.csv", "w");
    if (fp_all) {
        fprintf(fp_all, "cycle,rho,target,achieved");
        for (int i = 0; i < ECHO_NGYROS; ++i)
            if (gyroen.gyroEnable[i])
                fprintf(fp_all, ",gyro%d", i+1);
        fprintf(fp_all, "\n");
    }
    
    for (int cycle = 0; cycle < nCycles; ++cycle) {
        printf("\n--- PCS Cycle %d/%d ---\n", cycle + 1, nCycles);
        
        echo_optimizer(&scratch, &gyroen, &order, &params, &pslimits,
                       profileTarget, EchoProfileECH,
                       outAngles, outDutyCycles, &outFitness, &profiles);
        
        printf("Fitness: %.6f\n", outFitness);
        
        // Print angles for this cycle
        printf("Angles: ");
        for (int i = 0; i < ECHO_NGYROS; ++i)
            if (gyroen.gyroEnable[i])
                printf("G%d=%.0f° ", i+1, outAngles[i]);
        printf("\n");
        
        // Print duty cycles for this cycle
        printf("Duty Cycles: ");
        for (int i = 0; i < ECHO_NGYROPS; ++i)
            printf("PS%d=%.3f ", i+1, outDutyCycles[i]);
        printf("\n");
        
        // Write this cycle's profile to the all-cycles CSV
        if (fp_all) {
            for (int i = 0; i < NUM_ECHO_PCA_ROWS; ++i) {
                float rho = (float)i / (NUM_ECHO_PCA_ROWS - 1);
                float achieved = 0.0f;
                for (int j = 0; j < ECHO_NGYROS; ++j)
                    if (gyroen.gyroEnable[j])
                        achieved += profiles.best[j][i];
                
                fprintf(fp_all, "%d,%.6f,%.6f,%.6f", cycle + 1, rho, profiles.target[i], achieved);
                for (int j = 0; j < ECHO_NGYROS; ++j)
                    if (gyroen.gyroEnable[j])
                        fprintf(fp_all, ",%.6f", profiles.best[j][i]);
                fprintf(fp_all, "\n");
            }
        }
    }
    
    if (fp_all) {
        fclose(fp_all);
        printf("\n✓ All cycles written to profile_output_all_cycles.csv\n");
    }
    
    // Print final results
    printf("\n================================================\n");
    printf("FINAL RESULTS (Cycle %d)\n", nCycles);
    printf("================================================\n");
    printf("Best Fitness: %.6f\n\n", outFitness);
    
    printf("Gyrotron Angles:\n");
    for (int i = 0; i < ECHO_NGYROS; ++i)
        if (gyroen.gyroEnable[i])
            printf("  G%-2d: %6.2f deg\n", i+1, outAngles[i]);
    
    printf("\nPower Supply Duty Cycles:\n");
    for (int i = 0; i < ECHO_NGYROPS; ++i)
        printf("  PS%d: %.3f\n", i+1, outDutyCycles[i]);
    
    // Write gyrotron parameters to CSV
    FILE *fp_params = fopen("csv_files/gyro_parameters.csv", "w");
    if (fp_params) {
        fprintf(fp_params, "Gyrotron,Angle,DutyCycle,PowerSupply\n");
        for (int i = 0; i < ECHO_NGYROS; ++i) {
            if (gyroen.gyroEnable[i]) {
                int ps_idx = scratch.gyroPs[i];
                fprintf(fp_params, "G%d,%.2f,%.3f,PS%d\n", 
                       i+1, outAngles[i], outDutyCycles[ps_idx], ps_idx+1);
            }
        }
        fclose(fp_params);
        printf("\n✓ Parameters written to gyro_parameters.csv\n");
    }
    
    // Write profile to CSV
    FILE *fp = fopen("csv_files/profile_output.csv", "w");
    if (fp) {
        fprintf(fp, "rho,target,achieved");
        for (int i = 0; i < ECHO_NGYROS; ++i)
            if (gyroen.gyroEnable[i])
                fprintf(fp, ",gyro%d", i+1);
        fprintf(fp, "\n");
        
        for (int i = 0; i < NUM_ECHO_PCA_ROWS; ++i) {
            float rho = (float)i / (NUM_ECHO_PCA_ROWS - 1);
            float achieved = 0.0f;
            for (int j = 0; j < ECHO_NGYROS; ++j)
                if (gyroen.gyroEnable[j])
                    achieved += profiles.best[j][i];
            
            fprintf(fp, "%.6f,%.6f,%.6f", rho, profiles.target[i], achieved);
            for (int j = 0; j < ECHO_NGYROS; ++j)
                if (gyroen.gyroEnable[j])
                    fprintf(fp, ",%.6f", profiles.best[j][i]);
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("✓ Profile written to profile_output.csv\n");
    }
    
    printf("\n================================================\n");
    return 0;
}
