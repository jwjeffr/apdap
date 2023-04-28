import ovito
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class DefectTrajectory:

    position: np.ndarray
    displacement: np.ndarray
    frame: np.ndarray


def get_structure_dict():

    return {
        'OTHER': 0,
        'FCC': 1,
        'HCP': 2,
        'BCC': 3,
        'ICO': 4,
        'SC': 5,
        'CUBIC_DIAMOND': 6,
        'HEX_DIAMOND': 7,
        'GRAPHENE': 8
    }


@dataclass
class AddDefectModifier:

    def __call__(self, frame: int, data: ovito.data.DataCollection):

        position = data.particles['Position'][...]

        cell = data.cell[...]
        box_lengths = cell[:, 0:3].diagonal()

        mean_position = np.mean(position, axis=0)

        # need to check if cluster is split by boundary

        position_std = np.std(position, axis=0)
        threshold = 5.0
        split_dimensions = [position_std > threshold][0]

        for index, dim_bool in enumerate(split_dimensions):
            position_distribution = position[:, index].copy()
            if dim_bool:
                k_means = KMeans(n_clusters=2, n_init='auto').fit(position_distribution.reshape(-1, 1))
                centers = k_means.cluster_centers_
                labels = k_means.labels_
                largest_cluster_index = stats.mode(labels, keepdims=False).mode
                if centers[largest_cluster_index] < centers[largest_cluster_index - 1]:
                    modifier = -box_lengths[index]
                else:
                    modifier = box_lengths[index]
                for j, pos in enumerate(position_distribution):
                    prediction = k_means.predict(np.array(pos).reshape(-1, 1))
                    if prediction != largest_cluster_index:
                        position_distribution[index] += modifier
            mean_position[index] = np.mean(position_distribution)

        data.particles_.add_particle(mean_position)


def create_defect_pipeline(file_name, rmsd_cutoff=0.12):

    structure_dict = get_structure_dict()

    pipeline = ovito.io.import_file(file_name)
    pipeline.modifiers.append(ovito.modifiers.PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff))

    # calculate most common structure type
    initial_frame = pipeline.compute(0)
    structure = 'OTHER'

    for key, value in initial_frame.attributes.items():
        if 'PolyhedralTemplateMatching.counts' not in key:
            continue
        if initial_frame.attributes[f'PolyhedralTemplateMatching.counts.{structure}'] < value:
            _, __, structure = key.split('.')

    # find atoms where structure is not the most common structure

    modifiers = [
        ovito.modifiers.ExpressionSelectionModifier(expression=f'StructureType=={structure_dict[structure]}'),
        ovito.modifiers.DeleteSelectedModifier(),
        ovito.modifiers.ClusterAnalysisModifier(sort_by_size=True),
        ovito.modifiers.ExpressionSelectionModifier(expression='Cluster!=1'),
        ovito.modifiers.DeleteSelectedModifier(),
        AddDefectModifier(),
        ovito.modifiers.ExpressionSelectionModifier(expression='ParticleType!=0'),
        ovito.modifiers.DeleteSelectedModifier()
    ]
    for modifier in modifiers:
        pipeline.modifiers.append(modifier)

    pipeline.modifiers.append(ovito.modifiers.CalculateDisplacementsModifier())

    return pipeline


def create_combined_pipeline(file_name, rmsd_cutoff=0.12):

    unmodified_pipeline = ovito.io.import_file(file_name)
    pipeline = create_defect_pipeline(file_name, rmsd_cutoff)

    unmodified_pipeline.modifiers.append(ovito.modifiers.CombineDatasetsModifier(source=pipeline.data_provider))
    unmodified_pipeline.modifiers.append(ovito.modifiers.CalculateDisplacementsModifier())

    return unmodified_pipeline


def get_defect_trajectory(file_name):

    pipeline = create_defect_pipeline(file_name)

    position = np.zeros((pipeline.source.num_frames, 3))
    displacement = np.zeros((pipeline.source.num_frames, 3))
    frames = np.arange(pipeline.source.num_frames)

    for frame in frames:
        data = pipeline.compute(frame)
        position[frame] = data.particles['Position'][...].reshape((-1,))
        displacement[frame] = data.particles['Displacement'][...].reshape((-1,))

    return DefectTrajectory(position, displacement, frames)


def main():

    defect_trajectory = get_defect_trajectory('lennard_jonesV.dump')
    displacement = defect_trajectory.displacement
    time_steps = np.max(displacement.shape)

    square_displacement = np.linalg.norm(displacement, axis=1) ** 2
    plt.plot(time_steps, square_displacement)
    plt.savefig('test.png')


if __name__ == '__main__':

    main()
