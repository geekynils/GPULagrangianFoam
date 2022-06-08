// Getters

inline vec3 FlatMesh::getCellCentre(const int& cellLabel) const {

	return makeVec3(
		cellCentres.at(cellLabel*4),
		cellCentres.at(cellLabel*4 + 1),
		cellCentres.at(cellLabel*4 + 2)
	);
}

inline int FlatMesh::getNFaces(const int& cellLabel) const {

	return nFacesPerCell.at(cellLabel);
}

inline vec3 FlatMesh::getFaceCentre(const int& faceLabel) const {

	return makeVec3(
		faceCentres.at(faceLabel*4),
		faceCentres.at(faceLabel*4 + 1),
		faceCentres.at(faceLabel*4 + 2)
	);
}

inline vec3 FlatMesh::getFaceNormal(const int& faceLabel) const {

	return makeVec3(
		faceNormals.at(faceLabel*4),
		faceNormals.at(faceLabel*4 + 1),
		faceNormals.at(faceLabel*4 + 2)
	);
}

inline std::vector<int> FlatMesh::getFaceLabels(const int& cellLabel) const {

	std::vector<int> faceLabels;
	int nFaces = nFacesPerCell.at(cellLabel);
	int startIdx = faceLabelsIndex.at(cellLabel);

	for(int i=0; i<nFaces; i++) {
		faceLabels.push_back(faceLabelsPerCell.at(startIdx + i));
	}

	return faceLabels;
}
