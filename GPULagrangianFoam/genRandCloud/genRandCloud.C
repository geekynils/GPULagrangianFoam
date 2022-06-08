#include <ctime>

#include "fvCFD.H"
#include "gpuParticleCloud.H"
#include "Random.H"
#include "octree.H"
#include "octreeDataCell.H"

using namespace Foam;

label gpuParticle::instances;

int main(int argc, char *argv[])
{
	// Argument processing -----------------------------------------------------

	argList::validArgs.append("number of particle to generate");

	// Use an octree instead of just checking every cell.
    argList::validOptions.insert("octree", "");

	#include "setRootCase.H"

    int nParticles;
    IStringStream(args.additionalArgs()[0])() >> nParticles;
    bool useOctree = args.optionFound("octree");

    // Mesh and time -----------------------------------------------------------

	#include "createTime.H"
	#include "createMesh.H"

    gpuParticleCloud cloud(
    	mesh,
    	"defaultCloud",
    	true
    );

    treeBoundBox meshBb(mesh.points());

    // Build octree ------------------------------------------------------------

	// Calculate typical cell related size to shift bb by.
	scalar typDim = meshBb.avgDim()/(2.0*Foam::cbrt(scalar(mesh.nCells())));

	// It's Recommended that the bb for the octree is slightly larger than
	// the bb of the mesh.
	treeBoundBox bb (
		meshBb.min(),
		meshBb.max() + vector(typDim, typDim, typDim)
	);

	// Wrap indices and mesh information into helper object
	octreeDataCell shapes(mesh);

	octree<octreeDataCell> oc (
		bb,  		// overall bounding box
		shapes,     // all information needed to do checks on cells
		1,          // min. levels
		10.0,       // max. size of leaves
		10.0        // maximum ratio of cubes v.s. cells
	);

    // Initialize variables ----------------------------------------------------

    Random rand(time(0));

    point min = meshBb.min();
    point max = meshBb.max();

    // Max magnitude of u
    scalar umax = 2;
    scalar d = 0.1;

    scalar posx, posy, posz;
    label cellLabel;
    point pos;
    vector u;

    // Main loop ---------------------------------------------------------------
    // Random points are generated within the bb of the mesh and it's checked if
    // they are inside the mesh, if not another point is generated until one of
    // them is inside of the mesh.

    for(int i=0; i<nParticles; i++) {

    	if(i % 1000 == 0) printf("Generated %i particles.\n", i);

    	do {
    		posx = rand.scalar01();
    		posx *= max.x() - min.x();
    		posx += min.x();

    		posy = rand.scalar01();
    		posy *= max.y() - min.y();
    		posy += min.y();

    		posz = rand.scalar01();
    		posz *= max.z() - min.z();
    		posz += min.z();

    		pos = point(posx, posy, posz);

    		if(useOctree) {
    			cellLabel = oc.find(pos);
    		} else {
    			cellLabel = mesh.findCell(pos);
    		}

    	} while (cellLabel == -1);

		u = rand.vector01();
		u /= mag(u);
		u *= umax;

		// TODO Hope this is freed somewhere
    	gpuParticle *particle = new gpuParticle(cloud, pos, cellLabel, d, u);
    	cloud.addParticle(particle);
    }

    cloud.writeFields();

    //runTime.write();

    return 0;
}

