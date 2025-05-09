#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
using namespace std;

// Structure to store particle properties
struct Particle {
    double x, y, z;       // Position
    double vx, vy, vz;    // Velocity
    double fx, fy, fz;    // Force
};

class MDSimulation {
private:
    // Simulation parameters
    int N;                // Number of particles
    double L;             // Box size
    double rho;           // Density
    double dt;            // Time step
    double tau;           // Characteristic time
    double rc;            // Cutoff radius
    double potentialEnergy;
    double kineticEnergy;
    
    vector<Particle> particles;
    vector<vector<int>> neighborList;
    
    // Random number generators
    mt19937 gen;
    uniform_real_distribution<> uniform_dist;
    normal_distribution<> normal_dist;
    
    // Output files
    ofstream energyFile;
    ofstream temperatureFile;
    ofstream velocityDistFile;
    ofstream msdFile;
    
    // For MSD calculation
    vector<double> initialX, initialY, initialZ;
    
public:
    MDSimulation(double boxSize, double density, double timeStep, double characTime) 
        : L(boxSize), rho(density), dt(timeStep), tau(characTime),
          uniform_dist(-0.5, 0.5), normal_dist(0.0, 1.0) {
        
        // Calculate number of particles
        N = static_cast<int>(rho * L * L * L);
        
        // Set cutoff radius (standard for LJ is 2.5σ)
        rc = 2.5;
        
        // Initialize random number generator
        random_device rd;
        gen = mt19937(rd());
        
        // Open output files
        energyFile.open("energy_data.txt");
        temperatureFile.open("temperature_data.txt");
        velocityDistFile.open("velocity_dist.txt");
        msdFile.open("msd_data.txt");
        
        cout << "Initializing simulation with " << N << " particles" << endl;
    }
    
    ~MDSimulation() {
        // Close files
        energyFile.close();
        temperatureFile.close();
        velocityDistFile.close();
        msdFile.close();
    }
    
    void initialize() {
        particles.resize(N);
        neighborList.resize(N);
        
        // Place particles randomly, avoiding overlap
        placeParticles();
        
        // Initialize velocities (either Gaussian or uniform)
        initializeVelocities(true);  // true = Gaussian, false = uniform
        
        // Remove center of mass motion
        removeCOMMotion();
        
        // Store initial positions for MSD calculation
        initialX.resize(N);
        initialY.resize(N);
        initialZ.resize(N);
        
        for (int i = 0; i < N; i++) {
            initialX[i] = particles[i].x;
            initialY[i] = particles[i].y;
            initialZ[i] = particles[i].z;
        }
    }
    
    void placeParticles() {
        // Simple algorithm: try random positions until no overlap
        for (int i = 0; i < N; i++) {
            bool validPosition = false;
            double x, y, z;
            
            while (!validPosition) {
                // Random position in the box
                x = L * uniform_dist(gen);
                y = L * uniform_dist(gen);
                z = L * uniform_dist(gen);
                
                validPosition = true;
                
                // Check for overlap with existing particles
                for (int j = 0; j < i; j++) {
                    double dx = x - particles[j].x;
                    double dy = y - particles[j].y;
                    double dz = z - particles[j].z;
                    
                    // Apply minimum image convention
                    dx -= L * round(dx / L);
                    dy -= L * round(dy / L);
                    dz -= L * round(dz / L);
                    
                    double r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 < 0.9*0.9) {  // Minimum separation of 0.9
                        validPosition = false;
                        break;
                    }
                }
            }
            
            particles[i].x = x;
            particles[i].y = y;
            particles[i].z = z;
        }
    }
    
    void initializeVelocities(bool useGaussian) {
        for (int i = 0; i < N; i++) {
            if (useGaussian) {
                // Gaussian distribution with zero mean and unit variance
                particles[i].vx = normal_dist(gen);
                particles[i].vy = normal_dist(gen);
                particles[i].vz = normal_dist(gen);
            } else {
                // Uniform distribution between -0.5 and 0.5
                particles[i].vx = uniform_dist(gen);
                particles[i].vy = uniform_dist(gen);
                particles[i].vz = uniform_dist(gen);
            }
        }
    }
    
    void removeCOMMotion() {
        double vx_total = 0, vy_total = 0, vz_total = 0;
        
        // Compute center of mass velocity
        for (int i = 0; i < N; i++) {
            vx_total += particles[i].vx;
            vy_total += particles[i].vy;
            vz_total += particles[i].vz;
        }
        
        // Subtract COM velocity from each particle
        for (int i = 0; i < N; i++) {
            particles[i].vx -= vx_total / N;
            particles[i].vy -= vy_total / N;
            particles[i].vz -= vz_total / N;
        }
    }
    
    void buildNeighborList() {
        double skin = 0.3;  // Skin distance added to cutoff
        double rList = rc + skin;
        
        // Clear old neighbor lists
        for (auto& list : neighborList) {
            list.clear();
        }
        
        // Build new neighbor lists
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;
                
                // Apply minimum image convention
                dx -= L * round(dx / L);
                dy -= L * round(dy / L);
                dz -= L * round(dz / L);
                
                double r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 < rList*rList) {
                    neighborList[i].push_back(j);
                    neighborList[j].push_back(i);
                }
            }
        }
    }
    
    void computeForces() {
        // Reset forces
        for (int i = 0; i < N; i++) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }
        
        potentialEnergy = 0.0;
        
        // Compute forces using neighbor list
        for (int i = 0; i < N; i++) {
            for (const auto& j : neighborList[i]) {
                if (j > i) {  // Avoid double counting
                    double dx = particles[i].x - particles[j].x;
                    double dy = particles[i].y - particles[j].y;
                    double dz = particles[i].z - particles[j].z;
                    
                    // Apply minimum image convention
                    dx -= L * round(dx / L);
                    dy -= L * round(dy / L);
                    dz -= L * round(dz / L);
                    
                    double r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 < rc*rc) {
                        double r2i = 1.0 / r2;
                        double r6i = r2i * r2i * r2i;
                        double ff = 48.0 * r2i * r6i * (r6i - 0.5);
                        
                        // Force
                        double fx = ff * dx;
                        double fy = ff * dy;
                        double fz = ff * dz;
                        
                        // Apply force to particles
                        particles[i].fx += fx;
                        particles[i].fy += fy;
                        particles[i].fz += fz;
                        
                        particles[j].fx -= fx;
                        particles[j].fy -= fy;
                        particles[j].fz -= fz;
                        
                        // Calculate potential energy (LJ)
                        potentialEnergy += 4.0 * r6i * (r6i - 1.0);
                    }
                }
            }
        }
    }
    
    void velocityVerlet() {
        // First half of velocity verlet: v(t+dt/2) = v(t) + a(t)*dt/2
        for (int i = 0; i < N; i++) {
            particles[i].vx += 0.5 * dt * particles[i].fx;
            particles[i].vy += 0.5 * dt * particles[i].fy;
            particles[i].vz += 0.5 * dt * particles[i].fz;
            
            // Update positions: r(t+dt) = r(t) + v(t+dt/2)*dt
            particles[i].x += dt * particles[i].vx;
            particles[i].y += dt * particles[i].vy;
            particles[i].z += dt * particles[i].vz;
            
            // Apply periodic boundary conditions
            particles[i].x -= L * floor(particles[i].x / L);
            particles[i].y -= L * floor(particles[i].y / L);
            particles[i].z -= L * floor(particles[i].z / L);
        }
        
        // Recompute forces with new positions
        buildNeighborList();
        computeForces();
        
        // Second half of velocity verlet: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
        for (int i = 0; i < N; i++) {
            particles[i].vx += 0.5 * dt * particles[i].fx;
            particles[i].vy += 0.5 * dt * particles[i].fy;
            particles[i].vz += 0.5 * dt * particles[i].fz;
        }
    }
    
    void computeEnergy() {
        kineticEnergy = 0.0;
        
        for (int i = 0; i < N; i++) {
            double v2 = particles[i].vx*particles[i].vx + 
                        particles[i].vy*particles[i].vy + 
                        particles[i].vz*particles[i].vz;
            kineticEnergy += 0.5 * v2;
        }
    }
    
    double computeTemperature() {
        return 2.0 * kineticEnergy / (3.0 * N);
    }
    
    double computeMSD() {
        double msd = 0.0;
        
        for (int i = 0; i < N; i++) {
            double dx = particles[i].x - initialX[i];
            double dy = particles[i].y - initialY[i];
            double dz = particles[i].z - initialZ[i];
            
            // Account for periodic boundary conditions in displacement
            dx -= L * round(dx / L);
            dy -= L * round(dy / L);
            dz -= L * round(dz / L);
            
            msd += dx*dx + dy*dy + dz*dz;
        }
        
        return msd / N;
    }
    
    void saveVelocityDistribution(double simulationTime) {
        // Only save at specific times: 0τ, 50τ, 100τ
        if (abs(simulationTime) < 0.01 || 
            abs(simulationTime - 50.0) < 0.01 || 
            abs(simulationTime - 100.0) < 0.01) {
            
            velocityDistFile << "# Time = " << simulationTime << endl;
            
            for (int i = 0; i < N; i++) {
                velocityDistFile << particles[i].vx << " " 
                                 << particles[i].vy << " " 
                                 << particles[i].vz << endl;
            }
            
            velocityDistFile << endl << endl;  // Add separator
        }
    }
    
    void run(int numSteps) {
        // Initial force calculation
        buildNeighborList();
        computeForces();
        
        for (int step = 0; step < numSteps; step++) {
            // Perform integration step
            velocityVerlet();
            
            // Compute energies
            computeEnergy();
            
            // Calculate temperature
            double temperature = computeTemperature();
            
            // Calculate MSD
            double msd = computeMSD();
            
            // Current simulation time
            double currentTime = step * dt / tau;
            
            // Save data every 10 steps
            if (step % 10 == 0) {
                // Save energy data
                energyFile << currentTime << " " 
                           << potentialEnergy / N << " " 
                           << kineticEnergy / N << " " 
                           << (potentialEnergy + kineticEnergy) / N << endl;
                
                // Save temperature data
                temperatureFile << currentTime << " " << temperature << endl;
                
                // Save MSD data
                msdFile << currentTime << " " << msd << endl;
            }
            
            // Save velocity distribution at specific times
            saveVelocityDistribution(currentTime);
            
            // Progress indicator
            if (step % 1000 == 0) {
                cout << "Step " << step << "/" << numSteps 
                          << " (Time: " << currentTime << "τ)" << endl;
                cout << "  Energy: " << (potentialEnergy + kineticEnergy) / N 
                          << " (PE: " << potentialEnergy / N 
                          << ", KE: " << kineticEnergy / N << ")" << endl;
                cout << "  Temperature: " << temperature << endl;
                cout << "  MSD: " << msd << endl;
            }
        }
    }
};

int main() {
    // Parameters
    double L = 20.0;               // Box size
    double rho = 0.7;              // Density
    double tau = pow(1.0, 0.5);  // Characteristic time (m*σ²/kB*T)^1/2, using reduced units
    double dt = 0.001 * tau;       // Time step
    int numSteps = 100 * tau / dt; // Simulate for 100τ
    
    // Create and run simulation
    MDSimulation sim(L, rho, dt, tau);
    sim.initialize();
    sim.run(numSteps);
    
    cout << "Simulation complete. Output files generated." << endl;
    
    return 0;
}