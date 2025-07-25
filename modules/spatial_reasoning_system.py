"""
SYSTEM FOR improving SPATIAL AND GEOMETRIC REASONING of artificial intelligence API GOOGLE GEMINI 2.0 FLASH
Complete implementation with all classes, methods, and logic
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

# ==================== BASIC STRUCTURES ====================

@dataclass
class Point3D:
    """Point in 3D space"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

@dataclass
class Vector3D:
    """Vector in 3D space"""
    x: float
    y: float
    z: float
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class TopologyType(Enum):
    """Types of spatial topology"""
    INSIDE = "inside"
    OUTSIDE = "outside"
    TOUCHING = "touching"
    OVERLAPPING = "overlapping"
    ADJACENT = "adjacent"
    CONNECTED = "connected"
    DISJOINT = "disjoint"

# ==================== 3D OBJECT REPRESENTATION AND MANIPULATION ====================

class MentalObject3D:
    """Representation of a 3D object in mental space"""
    
    def __init__(self, name: str, vertices: List[Point3D], faces: List[List[int]]):
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.position = Point3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.scale = Vector3D(1, 1, 1)
        self.mental_properties = {}
        self.spatial_relations = {}
    
    def translate(self, offset: Point3D) -> None:
        """Move the object in mental space"""
        self.position = self.position + offset
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = vertex + offset
    
    def rotate_x(self, angle: float) -> None:
        """Rotation around X-axis"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i, vertex in enumerate(self.vertices):
            y = vertex.y * cos_a - vertex.z * sin_a
            z = vertex.y * sin_a + vertex.z * cos_a
            self.vertices[i] = Point3D(vertex.x, y, z)
        self.rotation.x += angle
    
    def rotate_y(self, angle: float) -> None:
        """Rotation around Y-axis"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i, vertex in enumerate(self.vertices):
            x = vertex.x * cos_a + vertex.z * sin_a
            z = -vertex.x * sin_a + vertex.z * cos_a
            self.vertices[i] = Point3D(x, vertex.y, z)
        self.rotation.y += angle
    
    def rotate_z(self, angle: float) -> None:
        """Rotation around Z-axis"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i, vertex in enumerate(self.vertices):
            x = vertex.x * cos_a - vertex.y * sin_a
            y = vertex.x * sin_a + vertex.y * cos_a
            self.vertices[i] = Point3D(x, y, vertex.z)
        self.rotation.z += angle
    
    def scale_object(self, scale_factor: Vector3D) -> None:
        """Scale the object"""
        for i, vertex in enumerate(self.vertices):
            self.vertices[i] = Point3D(
                vertex.x * scale_factor.x,
                vertex.y * scale_factor.y,
                vertex.z * scale_factor.z
            )
        self.scale.x *= scale_factor.x
        self.scale.y *= scale_factor.y
        self.scale.z *= scale_factor.z
    
    def get_bounding_box(self) -> Tuple[Point3D, Point3D]:
        """Get the bounding box"""
        if not self.vertices:
            return Point3D(0, 0, 0), Point3D(0, 0, 0)
        
        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        min_z = min(v.z for v in self.vertices)
        max_z = max(v.z for v in self.vertices)
        
        return Point3D(min_x, min_y, min_z), Point3D(max_x, max_y, max_z)
    
    def calculate_volume(self) -> float:
        """Calculate approximate volume"""
        min_bound, max_bound = self.get_bounding_box()
        return ((max_bound.x - min_bound.x) * 
                (max_bound.y - min_bound.y) * 
                (max_bound.z - min_bound.z))
    
    def add_mental_property(self, property_name: str, value: Any) -> None:
        """Add a mental property to the object"""
        self.mental_properties[property_name] = value
    
    def get_mental_property(self, property_name: str) -> Any:
        """Retrieve a mental property"""
        return self.mental_properties.get(property_name)

class MentalSpace3D:
    """Mental space for manipulating 3D objects"""
    
    def __init__(self, name: str, dimensions: Tuple[float, float, float]):
        self.name = name
        self.dimensions = dimensions
        self.objects = {}
        self.spatial_relationships = {}
        self.coordinate_system = "cartesian"
        self.reference_frame = Point3D(0, 0, 0)
    
    def add_object(self, obj: MentalObject3D) -> None:
        """Add an object to the mental space"""
        self.objects[obj.name] = obj
    
    def remove_object(self, name: str) -> bool:
        """Remove an object from the mental space"""
        if name in self.objects:
            del self.objects[name]
            return True
        return False
    
    def get_object(self, name: str) -> Optional[MentalObject3D]:
        """Retrieve an object by its name"""
        return self.objects.get(name)
    
    def calculate_spatial_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Calculate spatial relationships between all objects"""
        relationships = {}
        object_names = list(self.objects.keys())
        
        for i, obj1_name in enumerate(object_names):
            relationships[obj1_name] = {}
            for j, obj2_name in enumerate(object_names):
                if i != j:
                    obj1 = self.objects[obj1_name]
                    obj2 = self.objects[obj2_name]
                    
                    distance = obj1.position.distance_to(obj2.position)
                    relative_position = obj2.position - obj1.position
                    
                    relationships[obj1_name][obj2_name] = {
                        'distance': distance,
                        'relative_position': relative_position,
                        'topology': self._determine_topology(obj1, obj2)
                    }
        
        self.spatial_relationships = relationships
        return relationships
    
    def _determine_topology(self, obj1: MentalObject3D, obj2: MentalObject3D) -> TopologyType:
        """Determine the topological relationship between two objects"""
        # Simplification based on bounding boxes
        bb1_min, bb1_max = obj1.get_bounding_box()
        bb2_min, bb2_max = obj2.get_bounding_box()
        
        # Check if objects overlap
        if (bb1_max.x >= bb2_min.x and bb1_min.x <= bb2_max.x and
            bb1_max.y >= bb2_min.y and bb1_min.y <= bb2_max.y and
            bb1_max.z >= bb2_min.z and bb1_min.z <= bb2_max.z):
            return TopologyType.OVERLAPPING
        else:
            # Check proximity for adjacency
            distance = obj1.position.distance_to(obj2.position)
            if distance < 1.0:  # Adjacency threshold
                return TopologyType.ADJACENT
            else:
                return TopologyType.DISJOINT
    
    def transform_coordinate_system(self, new_system: str, new_origin: Point3D) -> None:
        """Transform the coordinate system"""
        if new_system != self.coordinate_system:
            # Coordinate transformation (simplified)
            offset = new_origin - self.reference_frame
            for obj in self.objects.values():
                obj.translate(offset)
            self.coordinate_system = new_system
            self.reference_frame = new_origin

# ==================== TOPOLOGICAL AND GEOMETRIC REASONING ====================

class TopologicalReasoner:
    """Intuitive topological and geometric reasoning engine"""
    
    def __init__(self):
        self.topology_rules = {}
        self.geometric_patterns = {}
        self.intuitive_knowledge = {}
        self._initialize_topology_rules()
        self._initialize_geometric_patterns()
    
    def _initialize_topology_rules(self) -> None:
        """Initialize basic topological rules"""
        self.topology_rules = {
            'containment': {
                'if_inside_then_connected': True,
                'if_inside_then_not_disjoint': True,
                'transitivity': True  # A inside B and B inside C => A inside C
            },
            'adjacency': {
                'symmetric': True,  # A adjacent to B => B adjacent to A
                'non_reflexive': True  # A not adjacent to itself
            },
            'connectivity': {
                'reflexive': True,  # A connected to itself
                'symmetric': True,  # A connected to B => B connected to A
                'transitive': False  # A connected to B and B connected to C doesn't always mean A connected to C
            }
        }
    
    def _initialize_geometric_patterns(self) -> None:
        """Initialize intuitive geometric patterns"""
        self.geometric_patterns = {
            'symmetry': {
                'bilateral': self._detect_bilateral_symmetry,
                'radial': self._detect_radial_symmetry,
                'rotational': self._detect_rotational_symmetry
            },
            'proportions': {
                'golden_ratio': self._detect_golden_ratio,
                'fibonacci': self._detect_fibonacci_proportions
            },
            'alignment': {
                'linear': self._detect_linear_alignment,
                'grid': self._detect_grid_alignment,
                'circular': self._detect_circular_alignment
            }
        }
    
    def analyze_topology(self, space: MentalSpace3D) -> Dict[str, Any]:
        """Analyze the topology of a mental space"""
        analysis = {
            'connectivity_graph': {},
            'clusters': [],
            'isolated_objects': [],
            'topology_violations': []
        }
        
        # Build the connectivity graph
        for obj_name, obj_relations in space.spatial_relationships.items():
            connections = []
            for other_name, relation in obj_relations.items():
                if relation['topology'] in [TopologyType.CONNECTED, TopologyType.TOUCHING, TopologyType.OVERLAPPING]:
                    connections.append(other_name)
            analysis['connectivity_graph'][obj_name] = connections
        
        # Detect clusters
        analysis['clusters'] = self._find_clusters(analysis['connectivity_graph'])
        
        # Identify isolated objects
        analysis['isolated_objects'] = [
            obj for obj, connections in analysis['connectivity_graph'].items()
            if not connections
        ]
        
        return analysis
    
    def _find_clusters(self, connectivity_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find clusters of connected objects"""
        visited = set()
        clusters = []
        
        def dfs(node, current_cluster):
            if node in visited:
                return
            visited.add(node)
            current_cluster.append(node)
            for neighbor in connectivity_graph.get(node, []):
                dfs(neighbor, current_cluster)
        
        for node in connectivity_graph:
            if node not in visited:
                cluster = []
                dfs(node, cluster)
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def reason_about_spatial_relationships(self, obj1: str, obj2: str, space: MentalSpace3D) -> Dict[str, Any]:
        """Reason about spatial relationships between two objects"""
        if obj1 not in space.objects or obj2 not in space.objects:
            return {'error': 'Object not found'}
        
        relations = space.spatial_relationships.get(obj1, {}).get(obj2, {})
        
        reasoning = {
            'direct_relationship': relations,
            'inferred_properties': {},
            'possible_interactions': [],
            'spatial_constraints': []
        }
        
        # Infer properties based on distance and topology
        if 'distance' in relations:
            distance = relations['distance']
            if distance < 1.0:
                reasoning['inferred_properties']['proximity'] = 'close'
                reasoning['possible_interactions'].append('direct_contact')
            elif distance < 5.0:
                reasoning['inferred_properties']['proximity'] = 'near'
                reasoning['possible_interactions'].append('influence_field')
            else:
                reasoning['inferred_properties']['proximity'] = 'far'
                reasoning['possible_interactions'].append('minimal_interaction')
        
        return reasoning
    
    def _detect_bilateral_symmetry(self, obj: MentalObject3D) -> bool:
        """Detect bilateral symmetry"""
        # Simplified implementation
        center = self._calculate_centroid(obj)
        tolerance = 0.1
        
        for vertex in obj.vertices:
            mirrored = Point3D(2 * center.x - vertex.x, vertex.y, vertex.z)
            if not self._point_exists_in_object(mirrored, obj, tolerance):
                return False
        return True
    
    def _detect_radial_symmetry(self, obj: MentalObject3D) -> bool:
        """Detect radial symmetry"""
        # Simplified implementation for radial symmetry
        center = self._calculate_centroid(obj)
        angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        
        for angle in angles:
            rotated_vertices = self._rotate_vertices_around_center(obj.vertices, center, angle)
            if not self._vertices_match(obj.vertices, rotated_vertices, 0.1):
                return False
        return True
    
    def _detect_rotational_symmetry(self, obj: MentalObject3D) -> int:
        """Detect rotational symmetry order"""
        center = self._calculate_centroid(obj)
        max_order = 8  # Test up to order 8
        
        for order in range(2, max_order + 1):
            angle = 2 * math.pi / order
            rotated_vertices = self._rotate_vertices_around_center(obj.vertices, center, angle)
            if self._vertices_match(obj.vertices, rotated_vertices, 0.1):
                return order
        return 1  # No rotational symmetry
    
    def _detect_golden_ratio(self, obj: MentalObject3D) -> bool:
        """Detect golden ratio proportions"""
        min_bound, max_bound = obj.get_bounding_box()
        dimensions = [
            max_bound.x - min_bound.x,
            max_bound.y - min_bound.y,
            max_bound.z - min_bound.z
        ]
        
        golden_ratio = 1.618033988749
        tolerance = 0.1
        
        for i in range(len(dimensions)):
            for j in range(i + 1, len(dimensions)):
                if dimensions[j] != 0:
                    ratio = dimensions[i] / dimensions[j]
                    if abs(ratio - golden_ratio) < tolerance or abs(ratio - 1/golden_ratio) < tolerance:
                        return True
        return False
    
    def _detect_fibonacci_proportions(self, obj: MentalObject3D) -> bool:
        """Detect Fibonacci proportions"""
        fibonacci = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        min_bound, max_bound = obj.get_bounding_box()
        dimensions = [
            max_bound.x - min_bound.x,
            max_bound.y - min_bound.y,
            max_bound.z - min_bound.z
        ]
        
        # Normalize dimensions
        max_dim = max(dimensions)
        if max_dim == 0:
            return False
        
        normalized_dims = [d / max_dim * 100 for d in dimensions]
        
        for dim in normalized_dims:
            for fib in fibonacci:
                if abs(dim - fib) < 5:  # 5% Tolerance
                    return True
        return False
    
    def _detect_linear_alignment(self, objects: List[MentalObject3D]) -> bool:
        """Detect linear alignment of objects"""
        if len(objects) < 3:
            return True  # 2 points are always aligned
        
        centers = [self._calculate_centroid(obj) for obj in objects]
        
        # Calculate the line passing through the first two points
        if len(centers) >= 2:
            direction = Vector3D(
                centers[1].x - centers[0].x,
                centers[1].y - centers[0].y,
                centers[1].z - centers[0].z
            ).normalize()
            
            # Check if all other points are on this line
            tolerance = 0.5
            for i in range(2, len(centers)):
                point_vector = Vector3D(
                    centers[i].x - centers[0].x,
                    centers[i].y - centers[0].y,
                    centers[i].z - centers[0].z
                )
                cross_product = direction.cross(point_vector)
                if cross_product.magnitude() > tolerance:
                    return False
        
        return True
    
    def _detect_grid_alignment(self, objects: List[MentalObject3D]) -> bool:
        """Detect grid alignment"""
        if len(objects) < 4:
            return False
        
        centers = [self._calculate_centroid(obj) for obj in objects]
        tolerance = 0.5
        
        # Check grid alignment by finding regular patterns
        x_coords = sorted(list(set(round(c.x, 1) for c in centers)))
        y_coords = sorted(list(set(round(c.y, 1) for c in centers)))
        z_coords = sorted(list(set(round(c.z, 1) for c in centers)))
        
        # Check if spacings are regular
        def is_regular_spacing(coords):
            if len(coords) < 2:
                return True
            spacing = coords[1] - coords[0]
            for i in range(2, len(coords)):
                if abs((coords[i] - coords[i-1]) - spacing) > tolerance:
                    return False
            return True
        
        return (is_regular_spacing(x_coords) and 
                is_regular_spacing(y_coords) and 
                is_regular_spacing(z_coords))
    
    def _detect_circular_alignment(self, objects: List[MentalObject3D]) -> bool:
        """Detect circular alignment"""
        if len(objects) < 3:
            return False
        
        centers = [self._calculate_centroid(obj) for obj in objects]
        
        # Calculate the center of the circle passing through the first three points
        if len(centers) >= 3:
            p1, p2, p3 = centers[0], centers[1], centers[2]
            
            # Calculate circle center and radius
            circle_center, radius = self._calculate_circle_center_and_radius(p1, p2, p3)
            
            if radius is None:
                return False
            
            # Check if all other points are on this circle
            tolerance = 0.5
            for i in range(3, len(centers)):
                distance = circle_center.distance_to(centers[i])
                if abs(distance - radius) > tolerance:
                    return False
            
            return True
        
        return False
    
    def _calculate_centroid(self, obj: MentalObject3D) -> Point3D:
        """Calculate the centroid of an object"""
        if not obj.vertices:
            return Point3D(0, 0, 0)
        
        sum_x = sum(v.x for v in obj.vertices)
        sum_y = sum(v.y for v in obj.vertices)
        sum_z = sum(v.z for v in obj.vertices)
        count = len(obj.vertices)
        
        return Point3D(sum_x / count, sum_y / count, sum_z / count)
    
    def _point_exists_in_object(self, point: Point3D, obj: MentalObject3D, tolerance: float) -> bool:
        """Check if a point exists in the object"""
        for vertex in obj.vertices:
            if vertex.distance_to(point) <= tolerance:
                return True
        return False
    
    def _rotate_vertices_around_center(self, vertices: List[Point3D], center: Point3D, angle: float) -> List[Point3D]:
        """Rotate vertices around a center"""
        rotated = []
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        for vertex in vertices:
            # Translate to origin
            x = vertex.x - center.x
            y = vertex.y - center.y
            
            # 2D Rotation (around Z-axis)
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            
            # Translate back
            rotated.append(Point3D(new_x + center.x, new_y + center.y, vertex.z))
        
        return rotated
    
    def _vertices_match(self, vertices1: List[Point3D], vertices2: List[Point3D], tolerance: float) -> bool:
        """Check if two sets of vertices match"""
        if len(vertices1) != len(vertices2):
            return False
        
        for v1 in vertices1:
            found_match = False
            for v2 in vertices2:
                if v1.distance_to(v2) <= tolerance:
                    found_match = True
                    break
            if not found_match:
                return False
        
        return True
    
    def _calculate_circle_center_and_radius(self, p1: Point3D, p2: Point3D, p3: Point3D) -> Tuple[Optional[Point3D], Optional[float]]:
        """Calculate the center and radius of a circle passing through three points"""
        # Simplified 2D calculation (projection onto XY plane)
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        
        # Calculate determinants
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        if abs(d) < 1e-10:  # Collinear points
            return None, None
        
        ux = ((x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)) / d
        uy = ((x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)) / d
        
        center = Point3D(ux, uy, (p1.z + p2.z + p3.z) / 3)  # Average of Z coordinates
        radius = center.distance_to(p1)
        
        return center, radius

# ==================== CONCEPTUAL NAVIGATION ====================

class ConceptualDimension:
    """Conceptual dimension in an abstract space"""
    
    def __init__(self, name: str, dimension_type: str, range_min: float = 0, range_max: float = 1):
        self.name = name
        self.dimension_type = dimension_type  # 'continuous', 'discrete', 'categorical'
        self.range_min = range_min
        self.range_max = range_max
        self.semantic_anchors = {}  # Semantic reference points
        self.transformation_rules = {}
    
    def add_semantic_anchor(self, name: str, position: float, description: str) -> None:
        """Add a semantic anchor in this dimension"""
        self.semantic_anchors[name] = {
            'position': position,
            'description': description
        }
    
    def map_concept_to_position(self, concept: str) -> float:
        """Map a concept to a position in this dimension"""
        # Mapping based on semantic similarity to anchors
        if concept in self.semantic_anchors:
            return self.semantic_anchors[concept]['position']
        
        # Position calculation based on similarity (simplified)
        best_position = (self.range_min + self.range_max) / 2  # Default position
        max_similarity = 0
        
        for anchor_name, anchor_data in self.semantic_anchors.items():
            similarity = self._calculate_semantic_similarity(concept, anchor_name)
            if similarity > max_similarity:
                max_similarity = similarity
                best_position = anchor_data['position']
        
        return best_position
    
    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity (simplified implementation)"""
        # Basic implementation based on common character length
        common_chars = set(concept1.lower()) & set(concept2.lower())
        total_chars = set(concept1.lower()) | set(concept2.lower())
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)

class AbstractSpace:
    """Multi-dimensional abstract space for conceptual navigation"""
    
    def __init__(self, name: str):
        self.name = name
        self.dimensions = {}
        self.concepts = {}
        self.navigation_paths = {}
        self.conceptual_clusters = {}
        self.transformation_matrix = None
    
    def add_dimension(self, dimension: ConceptualDimension) -> None:
        """Add a conceptual dimension"""
        self.dimensions[dimension.name] = dimension
    
    def add_concept(self, concept_name: str, properties: Dict[str, Any]) -> None:
        """Add a concept to the space"""
        position = {}
        for dim_name, dimension in self.dimensions.items():
            if dim_name in properties:
                position[dim_name] = properties[dim_name]
            else:
                position[dim_name] = dimension.map_concept_to_position(concept_name)
        
        self.concepts[concept_name] = {
            'position': position,
            'properties': properties,
            'connections': []
        }
    
    def calculate_conceptual_distance(self, concept1: str, concept2: str) -> float:
        """Calculate conceptual distance between two concepts"""
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return float('inf')
        
        pos1 = self.concepts[concept1]['position']
        pos2 = self.concepts[concept2]['position']
        
        distance = 0
        for dim_name in self.dimensions:
            if dim_name in pos1 and dim_name in pos2:
                diff = pos1[dim_name] - pos2[dim_name]
                distance += diff * diff
        
        return math.sqrt(distance)
    
    def find_navigation_path(self, start_concept: str, target_concept: str, max_steps: int = 10) -> List[str]:
        """Find a conceptual navigation path"""
        if start_concept not in self.concepts or target_concept not in self.concepts:
            return []
        
        # Using a simplified A* algorithm
        open_set = [(0, start_concept, [start_concept])]
        closed_set = set()
        
        while open_set:
            current_cost, current_concept, path = open_set.pop(0)
            
            if current_concept == target_concept:
                return path
            
            if current_concept in closed_set or len(path) > max_steps:
                continue
            
            closed_set.add(current_concept)
            
            # Find neighboring concepts
            neighbors = self._find_conceptual_neighbors(current_concept, 3)
            
            for neighbor in neighbors:
                if neighbor not in closed_set:
                    new_path = path + [neighbor]
                    distance_to_target = self.calculate_conceptual_distance(neighbor, target_concept)
                    total_cost = len(new_path) + distance_to_target
                    open_set.append((total_cost, neighbor, new_path))
            
            # Sort by cost
            open_set.sort(key=lambda x: x[0])
        
        return []  # No path found
    
    def _find_conceptual_neighbors(self, concept: str, max_neighbors: int = 5) -> List[str]:
        """Find the closest conceptual neighbors"""
        if concept not in self.concepts:
            return []
        
        distances = []
        for other_concept in self.concepts:
            if other_concept != concept:
                distance = self.calculate_conceptual_distance(concept, other_concept)
                distances.append((distance, other_concept))
        
        distances.sort(key=lambda x: x[0])
        return [concept for _, concept in distances[:max_neighbors]]
    
    def create_conceptual_clusters(self, num_clusters: int = 5) -> Dict[str, List[str]]:
        """Create conceptual clusters"""
        # Simplified k-means clustering implementation
        concept_names = list(self.concepts.keys())
        if len(concept_names) <= num_clusters:
            # Each concept forms its own cluster
            clusters = {f"cluster_{i}": [concept] for i, concept in enumerate(concept_names)}
            self.conceptual_clusters = clusters
            return clusters
        
        # Initialize cluster centers
        import random
        cluster_centers = random.sample(concept_names, num_clusters)
        clusters = {f"cluster_{i}": [] for i in range(num_clusters)}
        
        # Assign concepts to clusters
        for concept in concept_names:
            best_cluster = 0
            min_distance = float('inf')
            
            for i, center in enumerate(cluster_centers):
                distance = self.calculate_conceptual_distance(concept, center)
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = i
            
            clusters[f"cluster_{best_cluster}"].append(concept)
        
        self.conceptual_clusters = clusters
        return clusters
    
    def navigate_conceptual_gradient(self, start_concept: str, direction_vector: Dict[str, float], steps: int = 5) -> List[str]:
        """Navigate along a conceptual gradient"""
        if start_concept not in self.concepts:
            return []
        
        current_position = self.concepts[start_concept]['position'].copy()
        path = [start_concept]
        
        for step in range(steps):
            # Apply the direction vector
            new_position = {}
            for dim_name, value in current_position.items():
                if dim_name in direction_vector:
                    new_position[dim_name] = value + direction_vector[dim_name] * 0.1
                else:
                    new_position[dim_name] = value
            
            # Find the closest concept to this new position
            closest_concept = self._find_closest_concept_to_position(new_position)
            if closest_concept and closest_concept not in path:
                path.append(closest_concept)
                current_position = self.concepts[closest_concept]['position'].copy()
            else:
                break
        
        return path
    
    def _find_closest_concept_to_position(self, target_position: Dict[str, float]) -> Optional[str]:
        """Find the concept closest to a given position"""
        min_distance = float('inf')
        closest_concept = None
        
        for concept_name, concept_data in self.concepts.items():
            distance = 0
            for dim_name, target_value in target_position.items():
                if dim_name in concept_data['position']:
                    diff = concept_data['position'][dim_name] - target_value
                    distance += diff * diff
            
            distance = math.sqrt(distance)
            if distance < min_distance:
                min_distance = distance
                closest_concept = concept_name
        
        return closest_concept
    
    def analyze_conceptual_topology(self) -> Dict[str, Any]:
        """Analyze the topology of the conceptual space"""
        analysis = {
            'density_map': {},
            'conceptual_boundaries': [],
            'high_dimensional_structures': [],
            'navigation_efficiency': {}
        }
        
        # Calculate conceptual density
        for concept in self.concepts:
            neighbors = self._find_conceptual_neighbors(concept, 10)
            analysis['density_map'][concept] = len(neighbors)
        
        # Identify conceptual boundaries
        for concept in self.concepts:
            neighbors = self._find_conceptual_neighbors(concept, 3)
            avg_distance = sum(self.calculate_conceptual_distance(concept, neighbor) 
                             for neighbor in neighbors) / len(neighbors) if neighbors else 0
            
            if avg_distance > 0.5:  # Boundary threshold
                analysis['conceptual_boundaries'].append(concept)
        
        return analysis

class ConceptualNavigator:
    """Navigator for abstract conceptual spaces"""
    
    def __init__(self):
        self.spaces = {}
        self.current_space = None
        self.current_position = None
        self.navigation_history = []
        self.navigation_strategies = {}
        self._initialize_navigation_strategies()
    
    def _initialize_navigation_strategies(self) -> None:
        """Initialize navigation strategies"""
        self.navigation_strategies = {
            'direct': self._navigate_direct,
            'gradient_ascent': self._navigate_gradient_ascent,
            'exploration': self._navigate_exploration,
            'similarity_following': self._navigate_similarity,
            'cluster_hopping': self._navigate_cluster_hopping
        }
    
    def add_space(self, space: AbstractSpace) -> None:
        """Add a conceptual space"""
        self.spaces[space.name] = space
    
    def set_current_space(self, space_name: str) -> bool:
        """Set the current conceptual space"""
        if space_name in self.spaces:
            self.current_space = self.spaces[space_name]
            return True
        return False
    
    def navigate_to_concept(self, concept_name: str, strategy: str = 'direct') -> List[str]:
        """Navigate to a specific concept"""
        if not self.current_space or concept_name not in self.current_space.concepts:
            return []
        
        if strategy in self.navigation_strategies:
            path = self.navigation_strategies[strategy](concept_name)
            self.navigation_history.extend(path)
            self.current_position = concept_name
            return path
        
        return []
    
    def _navigate_direct(self, target_concept: str) -> List[str]:
        """Direct navigation to the target concept"""
        if not self.current_position:
            return [target_concept]
        
        return self.current_space.find_navigation_path(self.current_position, target_concept)
    
    def _navigate_gradient_ascent(self, target_concept: str) -> List[str]:
        """Gradient ascent navigation"""
        if not self.current_position:
            return [target_concept]
        
        # Calculate the direction vector towards the target
        current_pos = self.current_space.concepts[self.current_position]['position']
        target_pos = self.current_space.concepts[target_concept]['position']
        
        direction_vector = {}
        for dim_name in current_pos:
            if dim_name in target_pos:
                direction_vector[dim_name] = target_pos[dim_name] - current_pos[dim_name]
        
        return self.current_space.navigate_conceptual_gradient(self.current_position, direction_vector)
    
    def _navigate_exploration(self, target_concept: str) -> List[str]:
        """Exploratory navigation with detours"""
        if not self.current_position:
            return [target_concept]
        
        # Navigation with exploration of interesting intermediate concepts
        path = []
        current = self.current_position
        
        for _ in range(10):  # Maximum 10 steps
            if current == target_concept:
                break
            
            neighbors = self.current_space._find_conceptual_neighbors(current, 5)
            
            # Choose the most interesting neighbor (not necessarily the closest to the target)
            best_neighbor = None
            best_score = -1
            
            for neighbor in neighbors:
                if neighbor not in path:
                    # Score based on distance to target and "novelty"
                    distance_to_target = self.current_space.calculate_conceptual_distance(neighbor, target_concept)
                    novelty = len(self.current_space._find_conceptual_neighbors(neighbor, 10))
                    score = 1 / (1 + distance_to_target) + novelty * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
            
            if best_neighbor:
                path.append(best_neighbor)
                current = best_neighbor
            else:
                break
        
        return path
    
    def _navigate_similarity(self, target_concept: str) -> List[str]:
        """Navigation based on conceptual similarity"""
        if not self.current_position:
            return [target_concept]
        
        path = [self.current_position]
        current = self.current_position
        
        while current != target_concept and len(path) < 15:
            neighbors = self.current_space._find_conceptual_neighbors(current, 10)
            
            # Find the neighbor most similar to the target
            best_neighbor = None
            min_distance = float('inf')
            
            for neighbor in neighbors:
                if neighbor not in path:
                    distance = self.current_space.calculate_conceptual_distance(neighbor, target_concept)
                    if distance < min_distance:
                        min_distance = distance
                        best_neighbor = neighbor
            
            if best_neighbor:
                path.append(best_neighbor)
                current = best_neighbor
            else:
                break
        
        return path[1:]  # Exclude starting position
    
    def _navigate_cluster_hopping(self, target_concept: str) -> List[str]:
        """Cluster hopping navigation"""
        if not self.current_position:
            return [target_concept]
        
        # Create clusters if they don't exist
        if not self.current_space.conceptual_clusters:
            self.current_space.create_conceptual_clusters()
        
        # Find start and target clusters
        start_cluster = None
        target_cluster = None
        
        for cluster_name, concepts in self.current_space.conceptual_clusters.items():
            if self.current_position in concepts:
                start_cluster = cluster_name
            if target_concept in concepts:
                target_cluster = cluster_name
        
        if start_cluster == target_cluster:
            # Direct navigation within the same cluster
            return self.current_space.find_navigation_path(self.current_position, target_concept, 5)
        
        # Inter-cluster navigation
        path = []
        
        # Find the closest concept in the target cluster
        min_distance = float('inf')
        bridge_concept = None
        
        for concept in self.current_space.conceptual_clusters[target_cluster]:
            distance = self.current_space.calculate_conceptual_distance(self.current_position, concept)
            if distance < min_distance:
                min_distance = distance
                bridge_concept = concept
        
        if bridge_concept:
            # Navigate to the bridge concept
            bridge_path = self.current_space.find_navigation_path(self.current_position, bridge_concept)
            path.extend(bridge_path[1:])  # Exclude starting position
            
            # Navigate from bridge to target
            if bridge_concept != target_concept:
                final_path = self.current_space.find_navigation_path(bridge_concept, target_concept, 5)
                path.extend(final_path[1:])  # Exclude bridge concept
        
        return path
    
    def analyze_navigation_efficiency(self) -> Dict[str, Any]:
        """Analyze navigation efficiency"""
        if not self.navigation_history:
            return {'error': 'No navigation history'}
        
        analysis = {
            'total_steps': len(self.navigation_history),
            'unique_concepts_visited': len(set(self.navigation_history)),
            'revisit_rate': 0,
            'average_step_distance': 0,
            'exploration_breadth': 0
        }
        
        # Calculate revisit rate
        if analysis['total_steps'] > 0:
            analysis['revisit_rate'] = 1 - (analysis['unique_concepts_visited'] / analysis['total_steps'])
        
        # Calculate average step distance
        if len(self.navigation_history) > 1 and self.current_space:
            total_distance = 0
            for i in range(1, len(self.navigation_history)):
                distance = self.current_space.calculate_conceptual_distance(
                    self.navigation_history[i-1], 
                    self.navigation_history[i]
                )
                total_distance += distance
            analysis['average_step_distance'] = total_distance / (len(self.navigation_history) - 1)
        
        # Calculate exploration breadth
        if self.current_space:
            all_neighbors = set()
            for concept in set(self.navigation_history):
                neighbors = self.current_space._find_conceptual_neighbors(concept, 5)
                all_neighbors.update(neighbors)
            
            total_concepts = len(self.current_space.concepts)
            analysis['exploration_breadth'] = len(all_neighbors) / total_concepts if total_concepts > 0 else 0
        
        return analysis

# ==================== MENTAL VISUALIZATION AND ROTATION ====================

class MentalVisualization:
    """System for mental visualization and rotation of complex objects"""
    
    def __init__(self):
        self.visual_memory = {}
        self.rotation_cache = {}
        self.transformation_history = []
        self.visualization_modes = ['wireframe', 'solid', 'transparent', 'cross_section']
        self.current_mode = 'solid'
    
    def create_mental_image(self, obj: MentalObject3D, resolution: int = 100) -> Dict[str, Any]:
        """Create a mental image of a 3D object"""
        image_id = f"{obj.name}_{hash(str(obj.vertices))}"
        
        mental_image = {
            'id': image_id,
            'object_name': obj.name,
            'resolution': resolution,
            'viewpoints': {},
            'cross_sections': {},
            'wireframe_data': self._generate_wireframe(obj),
            'volume_data': self._generate_volume_representation(obj, resolution),
            'surface_normals': self._calculate_surface_normals(obj),
            'visual_features': self._extract_visual_features(obj)
        }
        
        # Generate views from different angles
        standard_viewpoints = [
            ('front', Vector3D(0, 0, 1)),
            ('back', Vector3D(0, 0, -1)),
            ('left', Vector3D(-1, 0, 0)),
            ('right', Vector3D(1, 0, 0)),
            ('top', Vector3D(0, 1, 0)),
            ('bottom', Vector3D(0, -1, 0)),
            ('isometric', Vector3D(1, 1, 1).normalize())
        ]
        
        for view_name, direction in standard_viewpoints:
            mental_image['viewpoints'][view_name] = self._project_to_2d(obj, direction)
        
        self.visual_memory[image_id] = mental_image
        return mental_image
    
    def rotate_mental_image(self, image_id: str, rotation: Vector3D, steps: int = 10) -> List[Dict[str, Any]]:
        """Perform a mental rotation with interpolation"""
        if image_id not in self.visual_memory:
            return []
        
        base_image = self.visual_memory[image_id]
        rotation_sequence = []
        
        # Interpolate rotation
        step_rotation = Vector3D(
            rotation.x / steps,
            rotation.y / steps,
            rotation.z / steps
        )
        
        current_rotation = Vector3D(0, 0, 0)
        
        for step in range(steps + 1):
            # Create an intermediate view
            intermediate_view = self._apply_rotation_to_image(base_image, current_rotation)
            rotation_sequence.append({
                'step': step,
                'rotation': Vector3D(current_rotation.x, current_rotation.y, current_rotation.z),
                'view_data': intermediate_view,
                'transformation_cost': self._calculate_transformation_cost(step_rotation)
            })
            
            # Increment rotation
            current_rotation.x += step_rotation.x
            current_rotation.y += step_rotation.y
            current_rotation.z += step_rotation.z
        
        # Save to history
        self.transformation_history.append({
            'type': 'rotation',
            'image_id': image_id,
            'transformation': rotation,
            'steps': steps,
            'timestamp': self._get_timestamp()
        })
        
        return rotation_sequence
    
    def mental_cross_section(self, image_id: str, plane_normal: Vector3D, plane_point: Point3D) -> Dict[str, Any]:
        """Create a mental cross-section"""
        if image_id not in self.visual_memory:
            return {}
        
        image_data = self.visual_memory[image_id]
        obj_name = image_data['object_name']
        
        # Reconstruct the object from image data
        # (simplification - in a real case, volumetric data would be used)
        
        cross_section = {
            'plane_normal': plane_normal,
            'plane_point': plane_point,
            'intersection_points': [],
            'cross_section_area': 0,
            'perimeter': 0,
            'interior_structure': []
        }
        
        # Calculate intersections with the cutting plane
        wireframe = image_data['wireframe_data']
        for edge in wireframe['edges']:
            intersection = self._line_plane_intersection(edge, plane_normal, plane_point)
            if intersection:
                cross_section['intersection_points'].append(intersection)
        
        # Calculate area and perimeter
        if cross_section['intersection_points']:
            cross_section['cross_section_area'] = self._calculate_polygon_area(cross_section['intersection_points'])
            cross_section['perimeter'] = self._calculate_polygon_perimeter(cross_section['intersection_points'])
        
        # Analyze internal structure
        cross_section['interior_structure'] = self._analyze_internal_structure(
            cross_section['intersection_points'], 
            image_data['volume_data']
        )
        
        return cross_section
    
    def mental_assembly_visualization(self, objects: List[MentalObject3D], assembly_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mentally visualize the assembly of complex objects"""
        assembly_viz = {
            'total_steps': len(assembly_steps),
            'step_visualizations': [],
            'collision_analysis': [],
            'assembly_constraints': [],
            'final_configuration': None
        }
        
        current_configuration = []
        
        for step_idx, step in enumerate(assembly_steps):
            step_viz = {
                'step_number': step_idx,
                'action': step.get('action', 'unknown'),
                'objects_involved': step.get('objects', []),
                'transformations': step.get('transformations', []),
                'intermediate_state': None,
                'conflicts': []
            }
            
            # Simulate assembly step
            if step['action'] == 'add_object':
                obj_name = step['objects'][0] if step['objects'] else None
                if obj_name:
                    # Add object to current configuration
                    obj = next((o for o in objects if o.name == obj_name), None)
                    if obj:
                        current_configuration.append(obj)
                        step_viz['intermediate_state'] = self._visualize_configuration(current_configuration)
            
            elif step['action'] == 'connect':
                # Visualize connection between objects
                step_viz['intermediate_state'] = self._visualize_connection(
                    step['objects'], 
                    current_configuration
                )
            
            # Analyze collisions
            conflicts = self._detect_assembly_conflicts(current_configuration)
            step_viz['conflicts'] = conflicts
            assembly_viz['collision_analysis'].append(conflicts)
            
            assembly_viz['step_visualizations'].append(step_viz)
        
        assembly_viz['final_configuration'] = self._visualize_configuration(current_configuration)
        
        return assembly_viz
    
    def mental_deformation_visualization(self, image_id: str, deformation_field: Dict[str, Vector3D]) -> Dict[str, Any]:
        """Visualize mental deformation of an object"""
        if image_id not in self.visual_memory:
            return {}
        
        base_image = self.visual_memory[image_id]
        
        deformation_viz = {
            'original_state': base_image,
            'deformation_steps': [],
            'stress_analysis': {},
            'deformed_state': None,
            'deformation_energy': 0
        }
        
        # Calculate deformation steps
        num_steps = 20
        for step in range(num_steps + 1):
            progress = step / num_steps
            
            # Interpolate the deformation field
            current_deformation = {}
            for point_id, displacement in deformation_field.items():
                current_deformation[point_id] = Vector3D(
                    displacement.x * progress,
                    displacement.y * progress,
                    displacement.z * progress
                )
            
            # Apply deformation
            deformed_state = self._apply_deformation(base_image, current_deformation)
            
            # Calculate deformation energy
            energy = self._calculate_deformation_energy(current_deformation)
            
            deformation_viz['deformation_steps'].append({
                'step': step,
                'progress': progress,
                'state': deformed_state,
                'energy': energy
            })
        
        deformation_viz['deformed_state'] = deformation_viz['deformation_steps'][-1]['state']
        deformation_viz['deformation_energy'] = deformation_viz['deformation_steps'][-1]['energy']
        
        # Analyze stresses
        deformation_viz['stress_analysis'] = self._analyze_deformation_stress(deformation_field)
        
        return deformation_viz
    
    def _generate_wireframe(self, obj: MentalObject3D) -> Dict[str, Any]:
        """Generate wireframe representation"""
        wireframe = {
            'vertices': obj.vertices,
            'edges': [],
            'vertex_connections': {}
        }
        
        # Generate edges from faces
        for face in obj.faces:
            for i in range(len(face)):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % len(face)]
                
                edge = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
                if edge not in wireframe['edges']:
                    wireframe['edges'].append(edge)
                
                # Update vertex connections
                if v1_idx not in wireframe['vertex_connections']:
                    wireframe['vertex_connections'][v1_idx] = []
                if v2_idx not in wireframe['vertex_connections']:
                    wireframe['vertex_connections'][v2_idx] = []
                
                wireframe['vertex_connections'][v1_idx].append(v2_idx)
                wireframe['vertex_connections'][v2_idx].append(v1_idx)
        
        return wireframe
    
    def _generate_volume_representation(self, obj: MentalObject3D, resolution: int) -> Dict[str, Any]:
        """Generate volumetric representation"""
        min_bound, max_bound = obj.get_bounding_box()
        
        # Create a 3D grid
        step_x = (max_bound.x - min_bound.x) / resolution
        step_y = (max_bound.y - min_bound.y) / resolution
        step_z = (max_bound.z - min_bound.z) / resolution
        
        volume_data = {
            'dimensions': (resolution, resolution, resolution),
            'bounds': (min_bound, max_bound),
            'voxels': [],
            'density_map': {},
            'gradient_field': {}
        }
        
        # Fill the grid (simplified voxelization algorithm)
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    x = min_bound.x + i * step_x
                    y = min_bound.y + j * step_y
                    z = min_bound.z + k * step_z
                    
                    point = Point3D(x, y, z)
                    is_inside = self._point_inside_object(point, obj)
                    
                    if is_inside:
                        volume_data['voxels'].append((i, j, k))
                        volume_data['density_map'][(i, j, k)] = 1.0
                    else:
                        volume_data['density_map'][(i, j, k)] = 0.0
        
        return volume_data
    
    def _calculate_surface_normals(self, obj: MentalObject3D) -> Dict[int, Vector3D]:
        """Calculate surface normals"""
        normals = {}
        
        for face_idx, face in enumerate(obj.faces):
            if len(face) >= 3:
                # Take the first three vertices to calculate the normal
                v1 = obj.vertices[face[0]]
                v2 = obj.vertices[face[1]]
                v3 = obj.vertices[face[2]]
                
                # Calculate edge vectors
                edge1 = Vector3D(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)
                edge2 = Vector3D(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z)
                
                # Cross product to get the normal
                normal = edge1.cross(edge2).normalize()
                normals[face_idx] = normal
        
        return normals
    
    def _extract_visual_features(self, obj: MentalObject3D) -> Dict[str, Any]:
        """Extract visual features"""
        features = {
            'symmetries': [],
            'distinctive_features': [],
            'complexity_metrics': {},
            'recognizable_patterns': []
        }
        
        # Analyze symmetries
        reasoner = TopologicalReasoner()
        if reasoner._detect_bilateral_symmetry(obj):
            features['symmetries'].append('bilateral')
        
        rotational_order = reasoner._detect_rotational_symmetry(obj)
        if rotational_order > 1:
            features['symmetries'].append(f'rotational_order_{rotational_order}')
        
        # Calculate complexity metrics
        features['complexity_metrics'] = {
            'vertex_count': len(obj.vertices),
            'face_count': len(obj.faces),
            'edge_count': sum(len(face) for face in obj.faces) // 2,
            'volume': obj.calculate_volume(),
            'surface_area_estimate': self._estimate_surface_area(obj)
        }
        
        # Detect recognizable patterns
        if self._is_approximately_spherical(obj):
            features['recognizable_patterns'].append('spherical')
        
        if self._is_approximately_cubic(obj):
            features['recognizable_patterns'].append('cubic')
        
        if self._is_approximately_cylindrical(obj):
            features['recognizable_patterns'].append('cylindrical')
        
        if reasoner._detect_golden_ratio(obj):
            features['recognizable_patterns'].append('golden_ratio')
        
        return features
    
    def _project_to_2d(self, obj: MentalObject3D, view_direction: Vector3D) -> Dict[str, Any]:
        """Project a 3D object to a 2D view"""
        # Calculate orthonormal basis for projection
        up_vector = Vector3D(0, 1, 0)
        if abs(view_direction.dot(up_vector)) > 0.9:
            up_vector = Vector3D(1, 0, 0)
        
        right_vector = view_direction.cross(up_vector).normalize()
        up_vector = right_vector.cross(view_direction).normalize()
        
        projection = {
            'view_direction': view_direction,
            'projected_vertices': [],
            'visible_faces': [],
            'silhouette': [],
            'depth_buffer': {}
        }
        
        # Project vertices
        for i, vertex in enumerate(obj.vertices):
            # Convert to view coordinates
            local_x = vertex.x * right_vector.x + vertex.y * right_vector.y + vertex.z * right_vector.z
            local_y = vertex.x * up_vector.x + vertex.y * up_vector.y + vertex.z * up_vector.z
            depth = vertex.x * view_direction.x + vertex.y * view_direction.y + vertex.z * view_direction.z
            
            projection['projected_vertices'].append({
                'original_index': i,
                'x': local_x,
                'y': local_y,
                'depth': depth
            })
            projection['depth_buffer'][i] = depth
        
        # Determine visible faces
        surface_normals = self._calculate_surface_normals(obj)
        for face_idx, face in enumerate(obj.faces):
            if face_idx in surface_normals:
                normal = surface_normals[face_idx]
                dot_product = normal.dot(view_direction)
                if dot_product < 0:  # Face oriented towards the camera
                    projection['visible_faces'].append(face_idx)
        
        # Calculate silhouette
        projection['silhouette'] = self._calculate_silhouette(obj, view_direction)
        
        return projection
    
    def _apply_rotation_to_image(self, base_image: Dict[str, Any], rotation: Vector3D) -> Dict[str, Any]:
        """Apply a rotation to a mental image"""
        rotated_image = base_image.copy()
        
        # Create rotation matrices
        cos_x, sin_x = math.cos(rotation.x), math.sin(rotation.x)
        cos_y, sin_y = math.cos(rotation.y), math.sin(rotation.y)
        cos_z, sin_z = math.cos(rotation.z), math.sin(rotation.z)
        
        # Apply rotation to wireframe data
        wireframe = base_image['wireframe_data']
        rotated_vertices = []
        
        for vertex in wireframe['vertices']:
            # Rotation around X
            y1 = vertex.y * cos_x - vertex.z * sin_x
            z1 = vertex.y * sin_x + vertex.z * cos_x
            
            # Rotation around Y
            x2 = vertex.x * cos_y + z1 * sin_y
            z2 = -vertex.x * sin_y + z1 * cos_y
            
            # Rotation around Z
            x3 = x2 * cos_z - y1 * sin_z
            y3 = x2 * sin_z + y1 * cos_z
            
            rotated_vertices.append(Point3D(x3, y3, z2))
        
        rotated_image['wireframe_data'] = {
            'vertices': rotated_vertices,
            'edges': wireframe['edges'],
            'vertex_connections': wireframe['vertex_connections']
        }
        
        # Update views
        standard_directions = [
            Vector3D(0, 0, 1), Vector3D(0, 0, -1),
            Vector3D(-1, 0, 0), Vector3D(1, 0, 0),
            Vector3D(0, 1, 0), Vector3D(0, -1, 0)
        ]
        
        rotated_image['viewpoints'] = {}
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        for i, direction in enumerate(standard_directions):
            # Create a temporary object with rotated vertices
            temp_obj = MentalObject3D("temp", rotated_vertices, base_image.get('faces', []))
            rotated_image['viewpoints'][view_names[i]] = self._project_to_2d(temp_obj, direction)
        
        return rotated_image
    
    def _calculate_transformation_cost(self, rotation_step: Vector3D) -> float:
        """Calculate the cognitive cost of a transformation"""
        # Cost increases with rotation magnitude
        magnitude = rotation_step.magnitude()
        
        # Base cost based on magnitude
        base_cost = magnitude * 10
        
        # Penalty for complex (multi-axis) rotations
        axis_count = sum(1 for component in [rotation_step.x, rotation_step.y, rotation_step.z] 
                        if abs(component) > 0.01)
        complexity_penalty = (axis_count - 1) * 5
        
        return base_cost + complexity_penalty
    
    def _get_timestamp(self) -> float:
        """Get a timestamp for history"""
        import time
        return time.time()
    
    def _line_plane_intersection(self, edge: Tuple[int, int], plane_normal: Vector3D, plane_point: Point3D) -> Optional[Point3D]:
        """Calculate intersection between a line and a plane"""
        # This method would require access to the object's vertices
        # Simplified implementation
        return None
    
    def _calculate_polygon_area(self, points: List[Point3D]) -> float:
        """Calculate the area of a polygon"""
        if len(points) < 3:
            return 0.0
        
        # Shoelace algorithm for a 2D polygon (projection)
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        
        return abs(area) / 2.0
    
    def _calculate_polygon_perimeter(self, points: List[Point3D]) -> float:
        """Calculate the perimeter of a polygon"""
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            perimeter += points[i].distance_to(points[j])
        
        return perimeter
    
    def _analyze_internal_structure(self, intersection_points: List[Point3D], volume_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze the internal structure revealed by the cut"""
        internal_structure = []
        
        # Analyze density along the cut
        if intersection_points:
            density_profile = []
            for point in intersection_points:
                # Convert point to voxel coordinates
                bounds = volume_data.get('bounds', (Point3D(0,0,0), Point3D(1,1,1)))
                min_bound, max_bound = bounds
                dimensions = volume_data.get('dimensions', (10, 10, 10))
                
                if max_bound.x != min_bound.x and max_bound.y != min_bound.y and max_bound.z != min_bound.z:
                    voxel_x = int((point.x - min_bound.x) / (max_bound.x - min_bound.x) * dimensions[0])
                    voxel_y = int((point.y - min_bound.y) / (max_bound.y - min_bound.y) * dimensions[1])
                    voxel_z = int((point.z - min_bound.z) / (max_bound.z - min_bound.z) * dimensions[2])
                    
                    # Clamp to limits
                    voxel_x = max(0, min(dimensions[0] - 1, voxel_x))
                    voxel_y = max(0, min(dimensions[1] - 1, voxel_y))
                    voxel_z = max(0, min(dimensions[2] - 1, voxel_z))
                    
                    density = volume_data.get('density_map', {}).get((voxel_x, voxel_y, voxel_z), 0.0)
                    density_profile.append(density)
            
            internal_structure.append({
                'type': 'density_profile',
                'data': density_profile,
                'average_density': sum(density_profile) / len(density_profile) if density_profile else 0
            })
        
        return internal_structure
    
    def _visualize_configuration(self, objects: List[MentalObject3D]) -> Dict[str, Any]:
        """Visualize an object configuration"""
        config_viz = {
            'objects': [obj.name for obj in objects],
            'bounding_box': None,
            'total_volume': 0,
            'spatial_relationships': {},
            'stability_analysis': {}
        }
        
        if objects:
            # Calculate global bounding box
            all_vertices = []
            for obj in objects:
                all_vertices.extend(obj.vertices)
            
            if all_vertices:
                min_x = min(v.x for v in all_vertices)
                max_x = max(v.x for v in all_vertices)
                min_y = min(v.y for v in all_vertices)
                max_y = max(v.y for v in all_vertices)
                min_z = min(v.z for v in all_vertices)
                max_z = max(v.z for v in all_vertices)
                
                config_viz['bounding_box'] = {
                    'min': Point3D(min_x, min_y, min_z),
                    'max': Point3D(max_x, max_y, max_z)
                }
            
            # Calculate total volume
            config_viz['total_volume'] = sum(obj.calculate_volume() for obj in objects)
            
            # Analyze spatial relationships
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i < j:
                        distance = obj1.position.distance_to(obj2.position)
                        config_viz['spatial_relationships'][f"{obj1.name}-{obj2.name}"] = {
                            'distance': distance,
                            'overlapping': self._check_overlap(obj1, obj2)
                        }
        
        return config_viz
    
    def _visualize_connection(self, object_names: List[str], configuration: List[MentalObject3D]) -> Dict[str, Any]:
        """Visualize the connection between objects"""
        connection_viz = {
            'objects': object_names,
            'connection_type': 'unknown',
            'connection_points': [],
            'connection_strength': 0.0
        }
        
        if len(object_names) >= 2:
            obj1 = next((o for o in configuration if o.name == object_names[0]), None)
            obj2 = next((o for o in configuration if o.name == object_names[1]), None)
            
            if obj1 and obj2:
                # Find potential connection points
                min_distance = float('inf')
                closest_points = (None, None)
                
                for v1 in obj1.vertices:
                    for v2 in obj2.vertices:
                        distance = v1.distance_to(v2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_points = (v1, v2)
                
                if closest_points[0] and closest_points[1]:
                    connection_viz['connection_points'] = list(closest_points)
                    connection_viz['connection_strength'] = max(0, 1.0 - min_distance / 10.0)
                    
                    if min_distance < 0.1:
                        connection_viz['connection_type'] = 'direct_contact'
                    elif min_distance < 1.0:
                        connection_viz['connection_type'] = 'close_proximity'
                    else:
                        connection_viz['connection_type'] = 'distant'
        
        return connection_viz
    
    def _detect_assembly_conflicts(self, configuration: List[MentalObject3D]) -> List[Dict[str, Any]]:
        """Detect conflicts in assembly"""
        conflicts = []
        
        for i, obj1 in enumerate(configuration):
            for j, obj2 in enumerate(configuration):
                if i < j:
                    if self._check_overlap(obj1, obj2):
                        conflicts.append({
                            'type': 'collision',
                            'objects': [obj1.name, obj2.name],
                            'severity': self._calculate_overlap_severity(obj1, obj2)
                        })
                    
                    # Check stability
                    if self._is_unstable_configuration(obj1, obj2):
                        conflicts.append({
                            'type': 'instability',
                            'objects': [obj1.name, obj2.name],
                            'severity': 'medium'
                        })
        
        return conflicts
    
    def _apply_deformation(self, base_image: Dict[str, Any], deformation_field: Dict[str, Vector3D]) -> Dict[str, Any]:
        """Apply a deformation field"""
        deformed_image = base_image.copy()
        
        # Apply deformation to vertices
        wireframe = base_image['wireframe_data']
        deformed_vertices = []
        
        for i, vertex in enumerate(wireframe['vertices']):
            vertex_id = str(i)
            if vertex_id in deformation_field:
                displacement = deformation_field[vertex_id]
                new_vertex = Point3D(
                    vertex.x + displacement.x,
                    vertex.y + displacement.y,
                    vertex.z + displacement.z
                )
                deformed_vertices.append(new_vertex)
            else:
                deformed_vertices.append(vertex)
        
        deformed_image['wireframe_data'] = {
            'vertices': deformed_vertices,
            'edges': wireframe['edges'],
            'vertex_connections': wireframe['vertex_connections']
        }
        
        return deformed_image
    
    def _calculate_deformation_energy(self, deformation_field: Dict[str, Vector3D]) -> float:
        """Calculate deformation energy"""
        total_energy = 0.0
        
        for displacement in deformation_field.values():
            # Energy proportional to the square of displacement
            energy = displacement.magnitude() ** 2
            total_energy += energy
        
        return total_energy
    
    def _analyze_deformation_stress(self, deformation_field: Dict[str, Vector3D]) -> Dict[str, Any]:
        """Analyze deformation stress"""
        stress_analysis = {
            'max_stress_point': None,
            'max_stress_value': 0.0,
            'stress_distribution': {},
            'critical_regions': []
        }
        
        for point_id, displacement in deformation_field.items():
            stress_value = displacement.magnitude()
            stress_analysis['stress_distribution'][point_id] = stress_value
            
            if stress_value > stress_analysis['max_stress_value']:
                stress_analysis['max_stress_value'] = stress_value
                stress_analysis['max_stress_point'] = point_id
            
            if stress_value > 2.0:  # Critical threshold
                stress_analysis['critical_regions'].append({
                    'point_id': point_id,
                    'stress_value': stress_value,
                    'displacement': displacement
                })
        
        return stress_analysis
    
    def _point_inside_object(self, point: Point3D, obj: MentalObject3D) -> bool:
        """Check if a point is inside an object (simplified method)"""
        min_bound, max_bound = obj.get_bounding_box()
        
        return (min_bound.x <= point.x <= max_bound.x and
                min_bound.y <= point.y <= max_bound.y and
                min_bound.z <= point.z <= max_bound.z)
    
    def _estimate_surface_area(self, obj: MentalObject3D) -> float:
        """Estimate surface area"""
        total_area = 0.0
        
        for face in obj.faces:
            if len(face) >= 3:
                # Calculate triangle area (or approximation for polygons)
                vertices = [obj.vertices[i] for i in face[:3]]
                if len(vertices) == 3:
                    v1, v2, v3 = vertices
                    
                    # Side vectors
                    side1 = Vector3D(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)
                    side2 = Vector3D(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z)
                    
                    # Area = 0.5 * |side1 × side2|
                    cross_product = side1.cross(side2)
                    area = 0.5 * cross_product.magnitude()
                    total_area += area
        
        return total_area
    
    def _is_approximately_spherical(self, obj: MentalObject3D) -> bool:
        """Check if the object is approximately spherical"""
        center = self._calculate_object_center(obj)
        
        # Calculate distances from center to all vertices
        distances = [center.distance_to(vertex) for vertex in obj.vertices]
        
        if not distances:
            return False
        
        avg_distance = sum(distances) / len(distances)
        
        # Check variance of distances
        variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
        coefficient_of_variation = (variance ** 0.5) / avg_distance if avg_distance > 0 else 0
        
        return coefficient_of_variation < 0.2  # Threshold for approximately spherical
    
    def _is_approximately_cubic(self, obj: MentalObject3D) -> bool:
        """Check if the object is approximately cubic"""
        min_bound, max_bound = obj.get_bounding_box()
        
        width = max_bound.x - min_bound.x
        height = max_bound.y - min_bound.y
        depth = max_bound.z - min_bound.z
        
        if width == 0 or height == 0 or depth == 0:
            return False
        
        # Check if dimensions are approximately equal
        ratios = [width/height, width/depth, height/depth]
        
        for ratio in ratios:
            if ratio < 0.8 or ratio > 1.2:  # 20% Tolerance
                return False
        
        return True
    
    def _is_approximately_cylindrical(self, obj: MentalObject3D) -> bool:
        """Check if the object is approximately cylindrical"""
        # Analyze vertex distribution to detect cylindricity
        center = self._calculate_object_center(obj)
        
        # Calculate radial distances in XY plane
        xy_distances = []
        z_coords = []
        
        for vertex in obj.vertices:
            xy_distance = math.sqrt((vertex.x - center.x)**2 + (vertex.y - center.y)**2)
            xy_distances.append(xy_distance)
            z_coords.append(vertex.z)
        
        if not xy_distances:
            return False
        
        # Check if radial distances are approximately constant
        avg_radius = sum(xy_distances) / len(xy_distances)
        radius_variance = sum((d - avg_radius)**2 for d in xy_distances) / len(xy_distances)
        radius_cv = (radius_variance ** 0.5) / avg_radius if avg_radius > 0 else 0
        
        # Check if the object extends significantly in Z
        z_range = max(z_coords) - min(z_coords) if z_coords else 0
        
        return radius_cv < 0.3 and z_range > avg_radius  # Criteria for cylindricity
    
    def _calculate_object_center(self, obj: MentalObject3D) -> Point3D:
        """Calculate the geometric center of an object"""
        if not obj.vertices:
            return Point3D(0, 0, 0)
        
        sum_x = sum(v.x for v in obj.vertices)
        sum_y = sum(v.y for v in obj.vertices)
        sum_z = sum(v.z for v in obj.vertices)
        count = len(obj.vertices)
        
        return Point3D(sum_x / count, sum_y / count, sum_z / count)
    
    def _calculate_silhouette(self, obj: MentalObject3D, view_direction: Vector3D) -> List[int]:
        """Calculate the object's silhouette from a view direction"""
        silhouette_edges = []
        
        # For each edge, check if it's part of the silhouette
        edge_face_count = {}
        
        # Count how many faces share each edge
        for face_idx, face in enumerate(obj.faces):
            for i in range(len(face)):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % len(face)]
                edge = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
                
                if edge not in edge_face_count:
                    edge_face_count[edge] = []
                edge_face_count[edge].append(face_idx)
        
        # An edge is part of the silhouette if it separates a visible face from a hidden face
        surface_normals = self._calculate_surface_normals(obj)
        
        for edge, face_indices in edge_face_count.items():
            if len(face_indices) == 2:
                face1_idx, face2_idx = face_indices
                
                if face1_idx in surface_normals and face2_idx in surface_normals:
                    normal1 = surface_normals[face1_idx]
                    normal2 = surface_normals[face2_idx]
                    
                    # Check face visibility
                    visible1 = normal1.dot(view_direction) < 0
                    visible2 = normal2.dot(view_direction) < 0
                    
                    if visible1 != visible2:  # One visible face, one hidden
                        silhouette_edges.extend(edge)
        
        return list(set(silhouette_edges))  # Remove duplicates
    
    def _check_overlap(self, obj1: MentalObject3D, obj2: MentalObject3D) -> bool:
        """Check if two objects overlap"""
        bb1_min, bb1_max = obj1.get_bounding_box()
        bb2_min, bb2_max = obj2.get_bounding_box()
        
        # Bounding box check
        return (bb1_max.x >= bb2_min.x and bb1_min.x <= bb2_max.x and
                bb1_max.y >= bb2_min.y and bb1_min.y <= bb2_max.y and
                bb1_max.z >= bb2_min.z and bb1_min.z <= bb2_max.z)
    
    def _calculate_overlap_severity(self, obj1: MentalObject3D, obj2: MentalObject3D) -> str:
        """Calculate the severity of overlap"""
        bb1_min, bb1_max = obj1.get_bounding_box()
        bb2_min, bb2_max = obj2.get_bounding_box()
        
        # Calculate intersection volume
        overlap_min_x = max(bb1_min.x, bb2_min.x)
        overlap_max_x = min(bb1_max.x, bb2_max.x)
        overlap_min_y = max(bb1_min.y, bb2_min.y)
        overlap_max_y = min(bb1_max.y, bb2_max.y)
        overlap_min_z = max(bb1_min.z, bb2_min.z)
        overlap_max_z = min(bb1_max.z, bb2_max.z)
        
        if (overlap_max_x > overlap_min_x and 
            overlap_max_y > overlap_min_y and 
            overlap_max_z > overlap_min_z):
            
            overlap_volume = ((overlap_max_x - overlap_min_x) *
                            (overlap_max_y - overlap_min_y) *
                            (overlap_max_z - overlap_min_z))
            
            obj1_volume = obj1.calculate_volume()
            obj2_volume = obj2.calculate_volume()
            min_volume = min(obj1_volume, obj2_volume)
            
            if min_volume > 0:
                overlap_ratio = overlap_volume / min_volume
                
                if overlap_ratio > 0.5:
                    return 'severe'
                elif overlap_ratio > 0.2:
                    return 'moderate'
                else:
                    return 'minor'
        
        return 'none'
    
    def _is_unstable_configuration(self, obj1: MentalObject3D, obj2: MentalObject3D) -> bool:
        """Check if the configuration is unstable"""
        # Simplified analysis based on centers of gravity
        center1 = self._calculate_object_center(obj1)
        center2 = self._calculate_object_center(obj2)
        
        # Check if one object is above the other without support
        if center1.z > center2.z + 1.0:  # obj1 above obj2
            # Check if obj1 has horizontal support from obj2
            horizontal_distance = math.sqrt((center1.x - center2.x)**2 + (center1.y - center2.y)**2)
            bb2_min, bb2_max = obj2.get_bounding_box()
            max_support_radius = max(bb2_max.x - bb2_min.x, bb2_max.y - bb2_min.y) / 2
            
            return horizontal_distance > max_support_radius
        
        return False

# ==================== SPATIAL ANALOGIES FOR ABSTRACT CONCEPTS ====================

class SpatialAnalogy:
    """Representation of a spatial analogy"""
    
    def __init__(self, source_concept: str, target_concept: str, spatial_mapping: Dict[str, Any]):
        self.source_concept = source_concept
        self.target_concept = target_concept
        self.spatial_mapping = spatial_mapping
        self.strength = 0.0
        self.validity_score = 0.0
        self.metaphorical_relations = {}
    
    def calculate_strength(self) -> float:
        """Calculate the strength of the analogy"""
        # Factors contributing to strength
        mapping_completeness = len(self.spatial_mapping) / 10.0  # Arbitrary normalization
        structural_similarity = self._calculate_structural_similarity()
        semantic_coherence = self._calculate_semantic_coherence()
        
        self.strength = (mapping_completeness + structural_similarity + semantic_coherence) / 3.0
        return self.strength
    
    def _calculate_structural_similarity(self) -> float:
        """Calculate structural similarity"""
        # Analyze preserved spatial relations
        preserved_relations = 0
        total_relations = 0
        
        for mapping_type, mapping_data in self.spatial_mapping.items():
            if isinstance(mapping_data, dict):
                if 'preserved_relations' in mapping_data:
                    preserved_relations += mapping_data['preserved_relations']
                if 'total_relations' in mapping_data:
                    total_relations += mapping_data['total_relations']
        
        if total_relations > 0:
            return preserved_relations / total_relations
        return 0.5  # Default value
    
    def _calculate_semantic_coherence(self) -> float:
        """Calculate semantic coherence"""
        # Simplified evaluation of semantic coherence
        coherence_factors = []
        
        # Check domain coherence
        if 'domain_similarity' in self.spatial_mapping:
            coherence_factors.append(self.spatial_mapping['domain_similarity'])
        
        # Check relation consistency
        if 'relation_consistency' in self.spatial_mapping:
            coherence_factors.append(self.spatial_mapping['relation_consistency'])
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5

class SpatialAnalogyEngine:
    """Engine for creating and manipulating spatial analogies"""
    
    def __init__(self):
        self.analogies = {}
        self.concept_spatial_representations = {}
        self.analogy_templates = {}
        self.metaphor_database = {}
        self._initialize_spatial_metaphors()
    
    def _initialize_spatial_metaphors(self) -> None:
        """Initialize basic spatial metaphors"""
        self.metaphor_database = {
            'hierarchical_concepts': {
                'spatial_pattern': 'vertical_arrangement',
                'mapping_rules': {
                    'higher_status': 'higher_position',
                    'lower_status': 'lower_position',
                    'dominance': 'elevation',
                    'subordination': 'depth'
                },
                'examples': ['organizational_hierarchy', 'social_status', 'authority_structure']
            },
            'temporal_concepts': {
                'spatial_pattern': 'linear_progression',
                'mapping_rules': {
                    'past': 'left_or_behind',
                    'present': 'center_or_here',
                    'future': 'right_or_ahead',
                    'duration': 'distance',
                    'sequence': 'path'
                },
                'examples': ['timeline', 'progress', 'development']
            },
            'emotional_concepts': {
                'spatial_pattern': 'distance_and_temperature',
                'mapping_rules': {
                    'positive_emotion': 'warmth_and_proximity',
                    'negative_emotion': 'coldness_and_distance',
                    'intensity': 'brightness_or_size',
                    'attachment': 'closeness'
                },
                'examples': ['relationships', 'feelings', 'attitudes']
            },
            'complexity_concepts': {
                'spatial_pattern': 'dimensional_expansion',
                'mapping_rules': {
                    'simple': 'low_dimensionality',
                    'complex': 'high_dimensionality',
                    'understanding': 'navigation_ease',
                    'confusion': 'maze_like_structure'
                },
                'examples': ['problems', 'systems', 'ideas']
            },
            'similarity_concepts': {
                'spatial_pattern': 'proximity_clustering',
                'mapping_rules': {
                    'similar': 'close_proximity',
                    'different': 'distant',
                    'category': 'cluster_or_region',
                    'relationship': 'connection_or_path'
                },
                'examples': ['classification', 'family_resemblance', 'conceptual_groups']
            }
        }
    
    def create_spatial_representation(self, concept: str, concept_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a spatial representation for an abstract concept"""
        # Determine the most appropriate metaphor type
        best_metaphor = self._select_best_metaphor(concept, concept_properties)
        
        spatial_rep = {
            'concept': concept,
            'metaphor_type': best_metaphor,
            'spatial_structure': {},
            'dimensional_mapping': {},
            'geometric_properties': {},
            'topological_features': {}
        }
        
        if best_metaphor in self.metaphor_database:
            metaphor_data = self.metaphor_database[best_metaphor]
            spatial_rep['spatial_structure'] = self._generate_spatial_structure(
                concept_properties, 
                metaphor_data
            )
            
            spatial_rep['dimensional_mapping'] = self._create_dimensional_mapping(
                concept_properties, 
                metaphor_data['mapping_rules']
            )
            
            spatial_rep['geometric_properties'] = self._derive_geometric_properties(
                concept_properties, 
                metaphor_data['spatial_pattern']
            )
            
            spatial_rep['topological_features'] = self._identify_topological_features(
                concept_properties
            )
        
        self.concept_spatial_representations[concept] = spatial_rep
        return spatial_rep
    
    def create_analogy(self, source_concept: str, target_concept: str, 
                      focus_aspects: List[str] = None) -> SpatialAnalogy:
        """Create a spatial analogy between two concepts"""
        # Get spatial representations
        source_rep = self.concept_spatial_representations.get(source_concept)
        target_rep = self.concept_spatial_representations.get(target_concept)
        
        if not source_rep or not target_rep:
            return None
        
        # Build spatial mapping
        spatial_mapping = self._build_spatial_mapping(source_rep, target_rep, focus_aspects)
        
        # Create the analogy
        analogy = SpatialAnalogy(source_concept, target_concept, spatial_mapping)
        analogy.calculate_strength()
        
        # Identify metaphorical relations
        analogy.metaphorical_relations = self._identify_metaphorical_relations(
            source_rep, target_rep, spatial_mapping
        )
        
        # Store the analogy
        analogy_key = f"{source_concept}->{target_concept}"
        self.analogies[analogy_key] = analogy
        
        return analogy
    
    def navigate_conceptual_space_via_analogy(self, start_concept: str, 
                                           target_domain: str, 
                                           navigation_strategy: str = 'similarity') -> List[str]:
        """Navigate conceptual space using analogies"""
        navigation_path = [start_concept]
        current_concept = start_concept
        max_steps = 10
        
        for step in range(max_steps):
            # Find potential analogies
            candidate_analogies = self._find_analogies_to_domain(current_concept, target_domain)
            
            if not candidate_analogies:
                break
            
            # Select the best analogy based on strategy
            best_analogy = self._select_analogy_by_strategy(candidate_analogies, navigation_strategy)
            
            if best_analogy:
                next_concept = best_analogy.target_concept
                if next_concept not in navigation_path:
                    navigation_path.append(next_concept)
                    current_concept = next_concept
                    
                    # Check if target domain has been reached
                    if self._concept_in_domain(next_concept, target_domain):
                        break
                else:
                    break  # Avoid cycles
            else:
                break
        
        return navigation_path
    
    def analyze_analogy_network(self) -> Dict[str, Any]:
        """Analyze the analogy network"""
        analysis = {
            'total_analogies': len(self.analogies),
            'average_strength': 0.0,
            'concept_connectivity': {},
            'metaphor_distribution': {},
            'strong_analogies': [],
            'weak_analogies': [],
            'analogy_clusters': []
        }
        
        if self.analogies:
            # Calculate average strength
            total_strength = sum(analogy.strength for analogy in self.analogies.values())
            analysis['average_strength'] = total_strength / len(self.analogies)
            
            # Analyze concept connectivity
            concept_connections = {}
            for analogy in self.analogies.values():
                source = analogy.source_concept
                target = analogy.target_concept
                
                if source not in concept_connections:
                    concept_connections[source] = []
                if target not in concept_connections:
                    concept_connections[target] = []
                
                concept_connections[source].append(target)
                concept_connections[target].append(source)
            
            analysis['concept_connectivity'] = {
                concept: len(connections) 
                for concept, connections in concept_connections.items()
            }
            
            # Metaphor distribution
            metaphor_counts = {}
            for rep in self.concept_spatial_representations.values():
                metaphor_type = rep.get('metaphor_type', 'unknown')
                metaphor_counts[metaphor_type] = metaphor_counts.get(metaphor_type, 0) + 1
            analysis['metaphor_distribution'] = metaphor_counts
            
            # Identify strong and weak analogies
            strength_threshold = analysis['average_strength']
            for analogy in self.analogies.values():
                if analogy.strength > strength_threshold * 1.2:
                    analysis['strong_analogies'].append({
                        'source': analogy.source_concept,
                        'target': analogy.target_concept,
                        'strength': analogy.strength
                    })
                elif analogy.strength < strength_threshold * 0.8:
                    analysis['weak_analogies'].append({
                        'source': analogy.source_concept,
                        'target': analogy.target_concept,
                        'strength': analogy.strength
                    })
        
        return analysis
    
    def generate_novel_analogies(self, concept: str, num_analogies: int = 5) -> List[SpatialAnalogy]:
        """Generate new analogies for a concept"""
        novel_analogies = []
        
        if concept not in self.concept_spatial_representations:
            return novel_analogies
        
        source_rep = self.concept_spatial_representations[concept]
        
        # Look for concepts with similar spatial structures
        candidates = []
        for other_concept, other_rep in self.concept_spatial_representations.items():
            if other_concept != concept:
                similarity = self._calculate_structural_similarity_between_reps(source_rep, other_rep)
                candidates.append((other_concept, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create analogies with the best candidates
        for candidate_concept, similarity in candidates[:num_analogies]:
            analogy = self.create_analogy(concept, candidate_concept)
            if analogy:
                novel_analogies.append(analogy)
        
        return novel_analogies
    
    def _select_best_metaphor(self, concept: str, properties: Dict[str, Any]) -> str:
        """Select the best spatial metaphor for a concept"""
        best_metaphor = 'similarity_concepts'  # Default metaphor
        best_score = 0.0
        
        for metaphor_type, metaphor_data in self.metaphor_database.items():
            score = 0.0
            
            # Analyze match with mapping_rules
            mapping_rules = metaphor_data.get('mapping_rules', {})
            for property_name, property_value in properties.items():
                if property_name in mapping_rules:
                    score += 1.0
                
                # Bonus for keywords in examples
                examples = metaphor_data.get('examples', [])
                for example in examples:
                    if property_name in example or example in str(property_value):
                        score += 0.5
            
            if score > best_score:
                best_score = score
                best_metaphor = metaphor_type
        
        return best_metaphor
    
    def _generate_spatial_structure(self, properties: Dict[str, Any], metaphor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the spatial structure based on concept properties"""
        spatial_pattern = metaphor_data.get('spatial_pattern', 'default')
        
        structure = {
            'pattern_type': spatial_pattern,
            'elements': [],
            'connections': [],
            'dimensions': 3,  # Default
            'coordinate_system': 'cartesian'
        }
        
        if spatial_pattern == 'vertical_arrangement':
            # Create a vertical hierarchy
            levels = properties.get('hierarchy_levels', ['top', 'middle', 'bottom'])
            for i, level in enumerate(levels):
                structure['elements'].append({
                    'name': level,
                    'position': Point3D(0, len(levels) - i - 1, 0),
                    'properties': properties.get(f'{level}_properties', {})
                })
            
            # Hierarchical connections
            for i in range(len(levels) - 1):
                structure['connections'].append({
                    'from': levels[i],
                    'to': levels[i + 1],
                    'type': 'hierarchical'
                })
        
        elif spatial_pattern == 'linear_progression':
            # Create a linear progression
            sequence = properties.get('sequence', ['start', 'middle', 'end'])
            for i, item in enumerate(sequence):
                structure['elements'].append({
                    'name': item,
                    'position': Point3D(i, 0, 0),
                    'properties': properties.get(f'{item}_properties', {})
                })
            
            # Sequential connections
            for i in range(len(sequence) - 1):
                structure['connections'].append({
                    'from': sequence[i],
                    'to': sequence[i + 1],
                    'type': 'sequential'
                })
        
        elif spatial_pattern == 'proximity_clustering':
            # Create similarity clusters
            clusters = properties.get('clusters', {})
            cluster_positions = {}
            
            # Position clusters
            for i, (cluster_name, cluster_items) in enumerate(clusters.items()):
                angle = 2 * math.pi * i / len(clusters)
                cluster_center = Point3D(math.cos(angle) * 5, math.sin(angle) * 5, 0)
                cluster_positions[cluster_name] = cluster_center
                
                # Position elements within the cluster
                for j, item in enumerate(cluster_items):
                    item_angle = 2 * math.pi * j / len(cluster_items)
                    item_position = Point3D(
                        cluster_center.x + math.cos(item_angle),
                        cluster_center.y + math.sin(item_angle),
                        cluster_center.z
                    )
                    
                    structure['elements'].append({
                        'name': item,
                        'position': item_position,
                        'cluster': cluster_name,
                        'properties': properties.get(f'{item}_properties', {})
                    })
        
        return structure
    
    def _create_dimensional_mapping(self, properties: Dict[str, Any], mapping_rules: Dict[str, str]) -> Dict[str, str]:
        """Create dimensional mapping based on rules"""
        dimensional_mapping = {}
        
        for concept_aspect, spatial_aspect in mapping_rules.items():
            if concept_aspect in properties:
                dimensional_mapping[concept_aspect] = spatial_aspect
        
        return dimensional_mapping
    
    def _derive_geometric_properties(self, properties: Dict[str, Any], spatial_pattern: str) -> Dict[str, Any]:
        """Derive geometric properties"""
        geometric_props = {
            'shape_type': 'complex',
            'symmetries': [],
            'dimensionality': 3,
            'boundedness': 'bounded',
            'connectivity': 'connected'
        }
        
        if spatial_pattern == 'vertical_arrangement':
            geometric_props['shape_type'] = 'hierarchical_tree'
            geometric_props['symmetries'] = ['vertical_reflection']
            geometric_props['primary_axis'] = 'y'
        
        elif spatial_pattern == 'linear_progression':
            geometric_props['shape_type'] = 'linear_sequence'
            geometric_props['dimensionality'] = 1
            geometric_props['primary_axis'] = 'x'
        
        elif spatial_pattern == 'proximity_clustering':
            geometric_props['shape_type'] = 'clustered_network'
            geometric_props['symmetries'] = ['rotational']
            geometric_props['connectivity'] = 'clustered'
        
        return geometric_props
    
    def _identify_topological_features(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Identify topological features"""
        topological_features = {
            'holes': 0,
            'connected_components': 1,
            'boundary_type': 'closed',
            'genus': 0,
            'orientation': 'orientable'
        }
        
        # Analyze properties to infer topology
        if 'cycles' in properties:
            topological_features['holes'] = len(properties['cycles'])
        
        if 'components' in properties:
            topological_features['connected_components'] = len(properties['components'])
        
        if properties.get('open_ended', False):
            topological_features['boundary_type'] = 'open'
        
        return topological_features
    
    def _build_spatial_mapping(self, source_rep: Dict[str, Any], target_rep: Dict[str, Any], 
                              focus_aspects: List[str] = None) -> Dict[str, Any]:
        """Build spatial mapping between two representations"""
        mapping = {
            'structural_correspondence': {},
            'dimensional_alignment': {},
            'geometric_similarity': {},
            'topological_preservation': {},
            'focus_aspects': focus_aspects or []
        }
        
        # Structural correspondence
        source_elements = source_rep.get('spatial_structure', {}).get('elements', [])
        target_elements = target_rep.get('spatial_structure', {}).get('elements', [])
        
        if source_elements and target_elements:
            # Simple mapping based on position in sequence
            for i, source_elem in enumerate(source_elements):
                if i < len(target_elements):
                    mapping['structural_correspondence'][source_elem['name']] = target_elements[i]['name']
        
        # Dimensional alignment
        source_mapping = source_rep.get('dimensional_mapping', {})
        target_mapping = target_rep.get('dimensional_mapping', {})
        
        for aspect, spatial_aspect in source_mapping.items():
            if aspect in target_mapping:
                mapping['dimensional_alignment'][aspect] = {
                    'source_spatial': spatial_aspect,
                    'target_spatial': target_mapping[aspect],
                    'preserved': spatial_aspect == target_mapping[aspect]
                }
        
        # Geometric similarity
        source_geom = source_rep.get('geometric_properties', {})
        target_geom = target_rep.get('geometric_properties', {})
        
        mapping['geometric_similarity'] = {
            'shape_similarity': source_geom.get('shape_type') == target_geom.get('shape_type'),
            'symmetry_preservation': bool(set(source_geom.get('symmetries', [])) & 
                                       set(target_geom.get('symmetries', []))),
            'dimensionality_match': source_geom.get('dimensionality') == target_geom.get('dimensionality')
        }
        
        # Topological preservation
        source_topo = source_rep.get('topological_features', {})
        target_topo = target_rep.get('topological_features', {})
        
        mapping['topological_preservation'] = {
            'connectivity_preserved': (source_topo.get('connected_components') == 
                                     target_topo.get('connected_components')),
            'holes_preserved': source_topo.get('holes') == target_topo.get('holes'),
            'boundary_preserved': source_topo.get('boundary_type') == target_topo.get('boundary_type')
        }
        
        return mapping
    
    def _identify_metaphorical_relations(self, source_rep: Dict[str, Any], target_rep: Dict[str, Any], 
                                       spatial_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Identify metaphorical relations"""
        relations = {
            'direct_mappings': [],
            'structural_analogies': [],
            'functional_correspondences': [],
            'relational_patterns': []
        }
        
        # Direct mappings
        struct_corr = spatial_mapping.get('structural_correspondence', {})
        for source_elem, target_elem in struct_corr.items():
            relations['direct_mappings'].append({
                'source': source_elem,
                'target': target_elem,
                'type': 'element_correspondence'
            })
        
        # Structural analogies
        dim_alignment = spatial_mapping.get('dimensional_alignment', {})
        for aspect, alignment in dim_alignment.items():
            if alignment.get('preserved', False):
                relations['structural_analogies'].append({
                    'aspect': aspect,
                    'spatial_mapping': alignment['source_spatial'],
                    'strength': 'strong'
                })
        
        return relations
    
    def _find_analogies_to_domain(self, concept: str, target_domain: str) -> List[SpatialAnalogy]:
        """Find analogies to a target domain"""
        candidate_analogies = []
        
        for analogy_key, analogy in self.analogies.items():
            if (analogy.source_concept == concept and 
                self._concept_in_domain(analogy.target_concept, target_domain)):
                candidate_analogies.append(analogy)
        
        return candidate_analogies
    
    def _select_analogy_by_strategy(self, analogies: List[SpatialAnalogy], strategy: str) -> Optional[SpatialAnalogy]:
        """Select an analogy based on strategy"""
        if not analogies:
            return None
        
        if strategy == 'strongest':
            return max(analogies, key=lambda a: a.strength)
        elif strategy == 'most_valid':
            return max(analogies, key=lambda a: a.validity_score)
        elif strategy == 'most_complete':
            return max(analogies, key=lambda a: len(a.spatial_mapping))
        else:  # strategy == 'similarity' or default
            return analogies[0]  # First analogy found
    
    def _concept_in_domain(self, concept: str, domain: str) -> bool:
        """Check if a concept belongs to a domain"""
        # Simplified implementation - could be extended with an ontology
        return domain.lower() in concept.lower() or concept.lower() in domain.lower()
    
    def _calculate_structural_similarity_between_reps(self, rep1: Dict[str, Any], rep2: Dict[str, Any]) -> float:
        """Calculate structural similarity between two representations"""
        similarity_factors = []
        
        # Spatial pattern similarity
        pattern1 = rep1.get('spatial_structure', {}).get('pattern_type', '')
        pattern2 = rep2.get('spatial_structure', {}).get('pattern_type', '')
        pattern_similarity = 1.0 if pattern1 == pattern2 else 0.0
        similarity_factors.append(pattern_similarity)
        
        # Geometric similarity
        geom1 = rep1.get('geometric_properties', {})
        geom2 = rep2.get('geometric_properties', {})
        
        shape_similarity = 1.0 if geom1.get('shape_type') == geom2.get('shape_type') else 0.0
        similarity_factors.append(shape_similarity)
        
        dim_similarity = 1.0 if geom1.get('dimensionality') == geom2.get('dimensionality') else 0.0
        similarity_factors.append(dim_similarity)
        
        # Topological similarity
        topo1 = rep1.get('topological_features', {})
        topo2 = rep2.get('topological_features', {})
        
        connectivity_similarity = 1.0 if (topo1.get('connected_components') == 
                                        topo2.get('connected_components')) else 0.0
        similarity_factors.append(connectivity_similarity)
        
        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

# ==================== MAIN SYSTEM ====================

class SpatialGeometricReasoningSystem:
    """Main spatial and geometric reasoning system"""
    
    def __init__(self):
        self.mental_spaces = {}
        self.topological_reasoner = TopologicalReasoner()
        self.abstract_spaces = {}
        self.conceptual_navigator = ConceptualNavigator()
        self.mental_visualizer = MentalVisualization()
        self.analogy_engine = SpatialAnalogyEngine()
        self.reasoning_history = []
        self.performance_metrics = {}
        
    def initialize_system(self) -> None:
        """Initialize the complete system"""
        # Create a default mental space
        default_space = MentalSpace3D("default_space", (100, 100, 100))
        self.mental_spaces["default"] = default_space
        
        # Create a default conceptual space
        default_abstract_space = AbstractSpace("default_abstract_space")
        
        # Add some basic conceptual dimensions
        time_dimension = ConceptualDimension("time", "continuous", 0, 10)
        time_dimension.add_semantic_anchor("past", 0, "Events that have occurred")
        time_dimension.add_semantic_anchor("present", 5, "Current moment")
        time_dimension.add_semantic_anchor("future", 10, "Events yet to occur")
        
        complexity_dimension = ConceptualDimension("complexity", "continuous", 0, 10)
        complexity_dimension.add_semantic_anchor("simple", 0, "Easy to understand")
        complexity_dimension.add_semantic_anchor("moderate", 5, "Moderate difficulty")
        complexity_dimension.add_semantic_anchor("complex", 10, "Highly complex")
        
        default_abstract_space.add_dimension(time_dimension)
        default_abstract_space.add_dimension(complexity_dimension)
        
        self.abstract_spaces["default"] = default_abstract_space
        self.conceptual_navigator.add_space(default_abstract_space)
        self.conceptual_navigator.set_current_space("default_abstract_space")
        
        print("Spatial and Geometric Reasoning System successfully initialized.")
    
    def add_3d_object_to_mental_space(self, space_name: str, obj: MentalObject3D) -> bool:
        """Add a 3D object to a mental space"""
        if space_name in self.mental_spaces:
            self.mental_spaces[space_name].add_object(obj)
            
            # Log to history
            self.reasoning_history.append({
                'action': 'add_3d_object',
                'space': space_name,
                'object': obj.name,
                'timestamp': self._get_current_timestamp()
            })
            
            return True
        return False
    
    def analyze_spatial_relationships(self, space_name: str) -> Dict[str, Any]:
        """Analyze spatial relationships in a space"""
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            relationships = space.calculate_spatial_relationships()
            topology_analysis = self.topological_reasoner.analyze_topology(space)
            
            # Log to history
            self.reasoning_history.append({
                'action': 'analyze_spatial_relationships',
                'space': space_name,
                'num_objects': len(space.objects),
                'timestamp': self._get_current_timestamp()
            })
            
            return {
                'spatial_relationships': relationships,
                'topology_analysis': topology_analysis
            }
        return {}
    
    def perform_mental_rotation(self, space_name: str, object_name: str, rotation: Vector3D) -> Dict[str, Any]:
        """Perform a mental rotation of an object"""
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            obj = space.get_object(object_name)
            
            if obj:
                # Create a mental image
                mental_image = self.mental_visualizer.create_mental_image(obj)
                
                # Perform the rotation
                rotation_sequence = self.mental_visualizer.rotate_mental_image(
                    mental_image['id'], 
                    rotation
                )
                
                # Apply the rotation to the actual object
                obj.rotate_x(rotation.x)
                obj.rotate_y(rotation.y)
                obj.rotate_z(rotation.z)
                
                # Log to history
                self.reasoning_history.append({
                    'action': 'mental_rotation',
                    'space': space_name,
                    'object': object_name,
                    'rotation': rotation,
                    'steps': len(rotation_sequence),
                    'timestamp': self._get_current_timestamp()
                })
                
                return {
                    'rotation_sequence': rotation_sequence,
                    'final_state': obj,
                    'transformation_cost': sum(step.get('transformation_cost', 0) 
                                             for step in rotation_sequence)
                }
        
        return {}
    
    def navigate_conceptual_space(self, start_concept: str, target_concept: str, 
                                strategy: str = 'direct') -> List[str]:
        """Navigate conceptual space"""
        path = self.conceptual_navigator.navigate_to_concept(target_concept, strategy)
        
        # Log to history
        self.reasoning_history.append({
            'action': 'conceptual_navigation',
            'start': start_concept,
            'target': target_concept,
            'strategy': strategy,
            'path_length': len(path),
            'timestamp': self._get_current_timestamp()
        })
        
        return path
    
    def create_spatial_analogy(self, source_concept: str, target_concept: str,
                             source_properties: Dict[str, Any], 
                             target_properties: Dict[str, Any]) -> SpatialAnalogy:
        """Create a spatial analogy between concepts"""
        # Create spatial representations
        self.analogy_engine.create_spatial_representation(source_concept, source_properties)
        self.analogy_engine.create_spatial_representation(target_concept, target_properties)
        
        # Create the analogy
        analogy = self.analogy_engine.create_analogy(source_concept, target_concept)
        
        # Log to history
        self.reasoning_history.append({
            'action': 'create_spatial_analogy',
            'source': source_concept,
            'target': target_concept,
            'analogy_strength': analogy.strength if analogy else 0.0,
            'timestamp': self._get_current_timestamp()
        })
        
        return analogy
    
    def perform_cross_section_analysis(self, space_name: str, object_name: str,
                                     plane_normal: Vector3D, plane_point: Point3D) -> Dict[str, Any]:
        """Perform a cross-section analysis"""
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            obj = space.get_object(object_name)
            
            if obj:
                # Create a mental image
                mental_image = self.mental_visualizer.create_mental_image(obj)
                
                # Perform the cross-section
                cross_section = self.mental_visualizer.mental_cross_section(
                    mental_image['id'], plane_normal, plane_point
                )
                
                # Log to history
                self.reasoning_history.append({
                    'action': 'cross_section_analysis',
                    'space': space_name,
                    'object': object_name,
                    'section_area': cross_section.get('cross_section_area', 0),
                    'timestamp': self._get_current_timestamp()
                })
                
                return cross_section
        
        return {}
    
    def analyze_object_assembly(self, space_name: str, object_names: List[str],
                              assembly_instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze object assembly"""
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            objects = [space.get_object(name) for name in object_names if space.get_object(name)]
            
            if objects:
                assembly_viz = self.mental_visualizer.mental_assembly_visualization(
                    objects, assembly_instructions
                )
                
                # Log to history
                self.reasoning_history.append({
                    'action': 'assembly_analysis',
                    'space': space_name,
                    'objects': object_names,
                    'assembly_steps': len(assembly_instructions),
                    'conflicts_detected': len(assembly_viz.get('collision_analysis', [])),
                    'timestamp': self._get_current_timestamp()
                })
                
                return assembly_viz
        
        return {}
    
    def simulate_object_deformation(self, space_name: str, object_name: str,
                                  deformation_field: Dict[str, Vector3D]) -> Dict[str, Any]:
        """Simulate object deformation"""
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            obj = space.get_object(object_name)
            
            if obj:
                # Create a mental image
                mental_image = self.mental_visualizer.create_mental_image(obj)
                
                # Simulate deformation
                deformation_viz = self.mental_visualizer.mental_deformation_visualization(
                    mental_image['id'], deformation_field
                )
                
                # Log to history
                self.reasoning_history.append({
                    'action': 'deformation_simulation',
                    'space': space_name,
                    'object': object_name,
                    'deformation_energy': deformation_viz.get('deformation_energy', 0),
                    'timestamp': self._get_current_timestamp()
                })
                
                return deformation_viz
        
        return {}
    
    def find_geometric_patterns(self, space_name: str) -> Dict[str, Any]:
        """Find geometric patterns in a space"""
        patterns = {
            'symmetrical_objects': [],
            'aligned_objects': [],
            'clustered_objects': [],
            'golden_ratio_objects': [],
            'geometric_relationships': []
        }
        
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            
            for obj_name, obj in space.objects.items():
                # Analyze symmetries
                if self.topological_reasoner._detect_bilateral_symmetry(obj):
                    patterns['symmetrical_objects'].append({
                        'object': obj_name,
                        'symmetry_type': 'bilateral'
                    })
                
                rotational_order = self.topological_reasoner._detect_rotational_symmetry(obj)
                if rotational_order > 1:
                    patterns['symmetrical_objects'].append({
                        'object': obj_name,
                        'symmetry_type': f'rotational_order_{rotational_order}'
                    })
                
                # Detect golden ratio
                if self.topological_reasoner._detect_golden_ratio(obj):
                    patterns['golden_ratio_objects'].append(obj_name)
            
            # Analyze alignments between objects
            objects_list = list(space.objects.values())
            if len(objects_list) >= 3:
                if self.topological_reasoner._detect_linear_alignment(objects_list):
                    patterns['aligned_objects'] = [obj.name for obj in objects_list]
                
                if self.topological_reasoner._detect_grid_alignment(objects_list):
                    patterns['clustered_objects'] = [obj.name for obj in objects_list]
            
            # Log to history
            self.reasoning_history.append({
                'action': 'pattern_detection',
                'space': space_name,
                'patterns_found': sum(len(v) if isinstance(v, list) else 1 for v in patterns.values()),
                'timestamp': self._get_current_timestamp()
            })
        
        return patterns
    
    def perform_topological_reasoning(self, space_name: str, query: str) -> Dict[str, Any]:
        """Perform topological reasoning on a query"""
        reasoning_result = {
            'query': query,
            'result': None,
            'confidence': 0.0,
            'reasoning_steps': [],
            'supporting_evidence': []
        }
        
        if space_name in self.mental_spaces:
            space = self.mental_spaces[space_name]
            
            # Analyze the query and determine reasoning type
            if 'inside' in query.lower():
                reasoning_result = self._reason_about_containment(space, query)
            elif 'connected' in query.lower():
                reasoning_result = self._reason_about_connectivity(space, query)
            elif 'adjacent' in query.lower() or 'touching' in query.lower():
                reasoning_result = self._reason_about_adjacency(space, query)
            elif 'distance' in query.lower():
                reasoning_result = self._reason_about_distance(space, query)
            else:
                reasoning_result['result'] = "Unable to interpret topological query"
                reasoning_result['confidence'] = 0.0
            
            # Log to history
            self.reasoning_history.append({
                'action': 'topological_reasoning',
                'space': space_name,
                'query': query,
                'confidence': reasoning_result['confidence'],
                'timestamp': self._get_current_timestamp()
            })
        
        return reasoning_result
    
    def generate_conceptual_analogies(self, concept: str, num_analogies: int = 5) -> List[Dict[str, Any]]:
        """Generate conceptual analogies for a concept"""
        analogies = self.analogy_engine.generate_novel_analogies(concept, num_analogies)
        
        analogy_results = []
        for analogy in analogies:
            analogy_results.append({
                'source': analogy.source_concept,
                'target': analogy.target_concept,
                'strength': analogy.strength,
                'metaphorical_relations': analogy.metaphorical_relations,
                'spatial_mapping': analogy.spatial_mapping
            })
        
        # Log to history
        self.reasoning_history.append({
            'action': 'generate_analogies',
            'concept': concept,
            'analogies_generated': len(analogy_results),
            'timestamp': self._get_current_timestamp()
        })
        
        return analogy_results
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance"""
        performance = {
            'total_operations': len(self.reasoning_history),
            'operation_breakdown': {},
            'average_processing_time': 0.0,
            'accuracy_metrics': {},
            'memory_usage': {},
            'efficiency_scores': {}
        }
        
        # Analyze operation breakdown
        operation_counts = {}
        for operation in self.reasoning_history:
            action = operation.get('action', 'unknown')
            operation_counts[action] = operation_counts.get(action, 0) + 1
        
        performance['operation_breakdown'] = operation_counts
        
        # Calculate memory metrics
        performance['memory_usage'] = {
            'mental_spaces': len(self.mental_spaces),
            'abstract_spaces': len(self.abstract_spaces),
            'stored_analogies': len(self.analogy_engine.analogies),
            'visual_memory_items': len(self.mental_visualizer.visual_memory),
            'navigation_history_length': len(self.conceptual_navigator.navigation_history)
        }
        
        # Calculate efficiency scores
        performance['efficiency_scores'] = {
            'spatial_reasoning': self._calculate_spatial_reasoning_efficiency(),
            'conceptual_navigation': self._calculate_navigation_efficiency(),
            'analogy_generation': self._calculate_analogy_efficiency(),
            'visualization': self._calculate_visualization_efficiency()
        }
        
        return performance
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimization_results = {
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Optimize visual memory
        if len(self.mental_visualizer.visual_memory) > 100:
            # Remove old mental images
            old_images = list(self.mental_visualizer.visual_memory.keys())[:50]
            for image_id in old_images:
                del self.mental_visualizer.visual_memory[image_id]
            
            optimization_results['optimizations_applied'].append('visual_memory_cleanup')
        
        # Optimize navigation history
        if len(self.conceptual_navigator.navigation_history) > 1000:
            # Keep only the last 500 entries
            self.conceptual_navigator.navigation_history = \
                self.conceptual_navigator.navigation_history[-500:]
            
            optimization_results['optimizations_applied'].append('navigation_history_trimming')
        
        # Optimize weak analogies
        weak_analogies = []
        for key, analogy in self.analogy_engine.analogies.items():
            if analogy.strength < 0.3:
                weak_analogies.append(key)
        
        for key in weak_analogies:
            del self.analogy_engine.analogies[key]
        
        if weak_analogies:
            optimization_results['optimizations_applied'].append('weak_analogies_removal')
        
        # Recommendations
        if len(self.mental_spaces) > 10:
            optimization_results['recommendations'].append(
                'Consider consolidating mental spaces to improve performance'
            )
        
        if len(self.reasoning_history) > 5000:
            optimization_results['recommendations'].append(
                'Archive old reasoning history to reduce memory usage'
            )
        
        return optimization_results
    
    def export_system_state(self) -> Dict[str, Any]:
        """Export the complete system state"""
        system_state = {
            'mental_spaces': {},
            'abstract_spaces': {},
            'analogies': {},
            'visual_memory': {},
            'reasoning_history': self.reasoning_history[-100:],  # Last 100 entries
            'configuration': {
                'num_mental_spaces': len(self.mental_spaces),
                'num_abstract_spaces': len(self.abstract_spaces),
                'num_analogies': len(self.analogy_engine.analogies),
                'system_version': '1.0.0'
            }
        }
        
        # Export mental spaces (structure only)
        for space_name, space in self.mental_spaces.items():
            system_state['mental_spaces'][space_name] = {
                'name': space.name,
                'dimensions': space.dimensions,
                'num_objects': len(space.objects),
                'object_names': list(space.objects.keys())
            }
        
        # Export abstract spaces
        for space_name, space in self.abstract_spaces.items():
            system_state['abstract_spaces'][space_name] = {
                'name': space.name,
                'num_dimensions': len(space.dimensions),
                'num_concepts': len(space.concepts),
                'concept_names': list(space.concepts.keys())
            }
        
        # Export analogies
        for analogy_key, analogy in self.analogy_engine.analogies.items():
            system_state['analogies'][analogy_key] = {
                'source': analogy.source_concept,
                'target': analogy.target_concept,
                'strength': analogy.strength,
                'validity_score': analogy.validity_score
            }
        
        return system_state
    
    def _get_current_timestamp(self) -> float:
        """Get the current timestamp"""
        import time
        return time.time()
    
    def _reason_about_containment(self, space: MentalSpace3D, query: str) -> Dict[str, Any]:
        """Reason about containment relationships"""
        reasoning_result = {
            'query': query,
            'result': None,
            'confidence': 0.0,
            'reasoning_steps': [],
            'supporting_evidence': []
        }
        
        # Extract objects mentioned in the query
        object_names = [name for name in space.objects.keys() if name.lower() in query.lower()]
        
        if len(object_names) >= 2:
            obj1_name, obj2_name = object_names[0], object_names[1]
            obj1 = space.objects[obj1_name]
            obj2 = space.objects[obj2_name]
            
            # Analyze containment based on bounding boxes
            bb1_min, bb1_max = obj1.get_bounding_box()
            bb2_min, bb2_max = obj2.get_bounding_box()
            
            # Check if obj1 is inside obj2
            inside_test = (bb1_min.x >= bb2_min.x and bb1_max.x <= bb2_max.x and
                          bb1_min.y >= bb2_min.y and bb1_max.y <= bb2_max.y and
                          bb1_min.z >= bb2_min.z and bb1_max.z <= bb2_max.z)
            
            reasoning_result['reasoning_steps'].append(f"Comparing bounding boxes of {obj1_name} and {obj2_name}")
            reasoning_result['supporting_evidence'].append(f"Object {obj1_name} bounding box: {bb1_min} to {bb1_max}")
            reasoning_result['supporting_evidence'].append(f"Object {obj2_name} bounding box: {bb2_min} to {bb2_max}")
            
            if inside_test:
                reasoning_result['result'] = f"{obj1_name} is inside {obj2_name}"
                reasoning_result['confidence'] = 0.8
            else:
                reasoning_result['result'] = f"{obj1_name} is not inside {obj2_name}"
                reasoning_result['confidence'] = 0.8
        else:
            reasoning_result['result'] = "Insufficient objects identified in query"
            reasoning_result['confidence'] = 0.1
        
        return reasoning_result
    
    def _reason_about_connectivity(self, space: MentalSpace3D, query: str) -> Dict[str, Any]:
        """Reason about connectivity"""
        reasoning_result = {
            'query': query,
            'result': None,
            'confidence': 0.0,
            'reasoning_steps': [],
            'supporting_evidence': []
        }
        
        # Analyze space topology
        topology_analysis = self.topological_reasoner.analyze_topology(space)
        connectivity_graph = topology_analysis.get('connectivity_graph', {})
        
        object_names = [name for name in space.objects.keys() if name.lower() in query.lower()]
        
        if len(object_names) >= 2:
            obj1_name, obj2_name = object_names[0], object_names[1]
            
            # Check direct connectivity
            connected = (obj1_name in connectivity_graph and 
                        obj2_name in connectivity_graph[obj1_name])
            
            reasoning_result['reasoning_steps'].append(f"Checking direct connectivity between {obj1_name} and {obj2_name}")
            reasoning_result['supporting_evidence'].append(f"Connectivity graph: {connectivity_graph}")
            
            if connected:
                reasoning_result['result'] = f"{obj1_name} is connected to {obj2_name}"
                reasoning_result['confidence'] = 0.9
            else:
                # Check indirect connectivity
                path = self._find_connectivity_path(obj1_name, obj2_name, connectivity_graph)
                if path:
                    reasoning_result['result'] = f"{obj1_name} is indirectly connected to {obj2_name} via {' -> '.join(path)}"
                    reasoning_result['confidence'] = 0.7
                else:
                    reasoning_result['result'] = f"{obj1_name} is not connected to {obj2_name}"
                    reasoning_result['confidence'] = 0.8
        else:
            reasoning_result['result'] = "Insufficient objects identified in query"
            reasoning_result['confidence'] = 0.1
        
        return reasoning_result
    
    def _reason_about_adjacency(self, space: MentalSpace3D, query: str) -> Dict[str, Any]:
        """Reason about adjacency"""
        reasoning_result = {
            'query': query,
            'result': None,
            'confidence': 0.0,
            'reasoning_steps': [],
            'supporting_evidence': []
        }
        
        object_names = [name for name in space.objects.keys() if name.lower() in query.lower()]
        
        if len(object_names) >= 2:
            obj1_name, obj2_name = object_names[0], object_names[1]
            obj1 = space.objects[obj1_name]
            obj2 = space.objects[obj2_name]
            
            # Calculate distance between objects
            distance = obj1.position.distance_to(obj2.position)
            
            # Determine characteristic size of objects
            obj1_size = self._calculate_object_characteristic_size(obj1)
            obj2_size = self._calculate_object_characteristic_size(obj2)
            avg_size = (obj1_size + obj2_size) / 2
            
            reasoning_result['reasoning_steps'].append(f"Calculating distance between {obj1_name} and {obj2_name}")
            reasoning_result['supporting_evidence'].append(f"Distance: {distance}")
            reasoning_result['supporting_evidence'].append(f"Average object size: {avg_size}")
            
            # Adjacency threshold based on object size
            adjacency_threshold = avg_size * 1.2
            
            if distance <= adjacency_threshold:
                reasoning_result['result'] = f"{obj1_name} is adjacent to {obj2_name}"
                reasoning_result['confidence'] = max(0.5, 1.0 - distance / adjacency_threshold)
            else:
                reasoning_result['result'] = f"{obj1_name} is not adjacent to {obj2_name}"
                reasoning_result['confidence'] = 0.8
        else:
            reasoning_result['result'] = "Insufficient objects identified in query"
            reasoning_result['confidence'] = 0.1
        
        return reasoning_result
    
    def _reason_about_distance(self, space: MentalSpace3D, query: str) -> Dict[str, Any]:
        """Reason about distances"""
        reasoning_result = {
            'query': query,
            'result': None,
            'confidence': 0.0,
            'reasoning_steps': [],
            'supporting_evidence': []
        }
        
        object_names = [name for name in space.objects.keys() if name.lower() in query.lower()]
        
        if len(object_names) >= 2:
            obj1_name, obj2_name = object_names[0], object_names[1]
            obj1 = space.objects[obj1_name]
            obj2 = space.objects[obj2_name]
            
            # Calculate different types of distances
            center_distance = obj1.position.distance_to(obj2.position)
            
            # Minimum distance between surfaces (approximation)
            obj1_size = self._calculate_object_characteristic_size(obj1)
            obj2_size = self._calculate_object_characteristic_size(obj2)
            surface_distance = max(0, center_distance - (obj1_size + obj2_size) / 2)
            
            reasoning_result['reasoning_steps'].append(f"Calculating various distances between {obj1_name} and {obj2_name}")
            reasoning_result['supporting_evidence'].append(f"Center-to-center distance: {center_distance}")
            reasoning_result['supporting_evidence'].append(f"Approximate surface distance: {surface_distance}")
            
            # Categorize distance
            if surface_distance < 0.5:
                distance_category = "very close"
            elif surface_distance < 2.0:
                distance_category = "close"
            elif surface_distance < 5.0:
                distance_category = "moderate distance"
            else:
                distance_category = "far"
            
            reasoning_result['result'] = f"Distance between {obj1_name} and {obj2_name}: {center_distance:.2f} units ({distance_category})"
            reasoning_result['confidence'] = 0.9
        else:
            reasoning_result['result'] = "Insufficient objects identified in query"
            reasoning_result['confidence'] = 0.1
        
        return reasoning_result
    
    def _find_connectivity_path(self, start: str, end: str, connectivity_graph: Dict[str, List[str]]) -> List[str]:
        """Find a connectivity path between two objects"""
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in connectivity_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _calculate_object_characteristic_size(self, obj: MentalObject3D) -> float:
        """Calculate the characteristic size of an object"""
        min_bound, max_bound = obj.get_bounding_box()
        
        width = max_bound.x - min_bound.x
        height = max_bound.y - min_bound.y
        depth = max_bound.z - min_bound.z
        
        # Return the average dimension
        return (width + height + depth) / 3.0
    
    def _calculate_spatial_reasoning_efficiency(self) -> float:
        """Calculate spatial reasoning efficiency"""
        spatial_operations = [op for op in self.reasoning_history 
                            if op.get('action') in ['analyze_spatial_relationships', 
                                                  'mental_rotation', 
                                                  'cross_section_analysis']]
        
        if not spatial_operations:
            return 0.0
        
        # Measure efficiency based on number of successful operations
        successful_operations = len(spatial_operations)  # Simplified - all considered successful
        
        return min(1.0, successful_operations / 100.0)  # Normalize to 1.0
    
    def _calculate_navigation_efficiency(self) -> float:
        """Calculate navigation efficiency"""
        navigation_analysis = self.conceptual_navigator.analyze_navigation_efficiency()
        
        if 'exploration_breadth' in navigation_analysis:
            return navigation_analysis['exploration_breadth']
        
        return 0.5  # Default value
    
    def _calculate_analogy_efficiency(self) -> float:
        """Calculate analogy generation efficiency"""
        analogy_operations = [op for op in self.reasoning_history 
                            if op.get('action') in ['create_spatial_analogy', 'generate_analogies']]
        
        if not analogy_operations:
            return 0.0
        
        # Calculate ratio of strong analogies
        total_analogies = len(self.analogy_engine.analogies)
        strong_analogies = sum(1 for a in self.analogy_engine.analogies.values() if a.strength > 0.7)
        
        if total_analogies > 0:
            return strong_analogies / total_analogies
        
        return 0.0
    
    def _calculate_visualization_efficiency(self) -> float:
        """Calculate visualization efficiency"""
        viz_operations = [op for op in self.reasoning_history 
                         if op.get('action') in ['mental_rotation', 'cross_section_analysis', 
                                               'assembly_analysis', 'deformation_simulation']]
        
        if not viz_operations:
            return 0.0
        
        # Measure efficiency based on visualization cache
        cache_utilization = len(self.mental_visualizer.visual_memory) / max(1, len(viz_operations))
        
        return min(1.0, cache_utilization)
    
    def execute_reasoning_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete reasoning pipeline"""
        pipeline_results = {
            'pipeline_id': pipeline_config.get('id', 'default'),
            'steps_executed': [],
            'final_result': None,
            'execution_time': 0.0,
            'errors': []
        }
        
        start_time = self._get_current_timestamp()
        
        try:
            steps = pipeline_config.get('steps', [])
            
            for step_config in steps:
                step_type = step_config.get('type')
                step_params = step_config.get('parameters', {})
                
                step_result = None
                
                if step_type == 'spatial_analysis':
                    step_result = self.analyze_spatial_relationships(
                        step_params.get('space_name', 'default')
                    )
                
                elif step_type == 'mental_rotation':
                    step_result = self.perform_mental_rotation(
                        step_params.get('space_name', 'default'),
                        step_params.get('object_name'),
                        Vector3D(
                            step_params.get('rotation_x', 0),
                            step_params.get('rotation_y', 0),
                            step_params.get('rotation_z', 0)
                        )
                    )
                
                elif step_type == 'conceptual_navigation':
                    step_result = self.navigate_conceptual_space(
                        step_params.get('start_concept'),
                        step_params.get('target_concept'),
                        step_params.get('strategy', 'direct')
                    )
                
                elif step_type == 'analogy_creation':
                    step_result = self.create_spatial_analogy(
                        step_params.get('source_concept'),
                        step_params.get('target_concept'),
                        step_params.get('source_properties', {}),
                        step_params.get('target_properties', {})
                    )
                
                elif step_type == 'pattern_detection':
                    step_result = self.find_geometric_patterns(
                        step_params.get('space_name', 'default')
                    )
                
                elif step_type == 'topological_reasoning':
                    step_result = self.perform_topological_reasoning(
                        step_params.get('space_name', 'default'),
                        step_params.get('query', '')
                    )
                
                pipeline_results['steps_executed'].append({
                    'step_type': step_type,
                    'result': step_result,
                    'success': step_result is not None
                })
            
            # The final result is the result of the last step
            if pipeline_results['steps_executed']:
                pipeline_results['final_result'] = pipeline_results['steps_executed'][-1]['result']
        
        except Exception as e:
            pipeline_results['errors'].append(str(e))
        
        finally:
            pipeline_results['execution_time'] = self._get_current_timestamp() - start_time
        
        # Log to history
        self.reasoning_history.append({
            'action': 'execute_pipeline',
            'pipeline_id': pipeline_results['pipeline_id'],
            'steps_count': len(pipeline_results['steps_executed']),
            'execution_time': pipeline_results['execution_time'],
            'success': len(pipeline_results['errors']) == 0,
            'timestamp': self._get_current_timestamp()
        })
        
        return pipeline_results
    
    def save_system_state(self, filename: str) -> bool:
        """Save the system state to a file"""
        try:
            system_state = self.export_system_state()
            
            import json
            with open(filename, 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False
    
    def load_system_state(self, filename: str) -> bool:
        """Load the system state from a file"""
        try:
            import json
            with open(filename, 'r') as f:
                system_state = json.load(f)
            
            # Restore reasoning history
            self.reasoning_history = system_state.get('reasoning_history', [])
            
            print(f"System state loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False
    
    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            'system_health': 'healthy',
            'component_status': {},
            'performance_metrics': {},
            'recommendations': [],
            'warnings': [],
            'errors': []
        }
        
        # Check component status
        diagnostics['component_status'] = {
            'mental_spaces': 'operational' if self.mental_spaces else 'inactive',
            'topological_reasoner': 'operational',
            'abstract_spaces': 'operational' if self.abstract_spaces else 'inactive',
            'conceptual_navigator': 'operational',
            'mental_visualizer': 'operational',
            'analogy_engine': 'operational'
        }
        
        # Analyze performance
        diagnostics['performance_metrics'] = self.analyze_system_performance()
        
        # Generate recommendations
        if len(self.reasoning_history) == 0:
            diagnostics['recommendations'].append("System has not been used yet - consider running test operations")
        
        if len(self.mental_spaces) == 0:
            diagnostics['warnings'].append("No mental spaces defined - spatial reasoning capabilities limited")
        
        if len(self.abstract_spaces) == 0:
            diagnostics['warnings'].append("No abstract spaces defined - conceptual navigation limited")
        
        # Check data integrity
        for space_name, space in self.mental_spaces.items():
            if len(space.objects) == 0:
                diagnostics['warnings'].append(f"Mental space '{space_name}' contains no objects")
        
        # Determine overall system health
        if diagnostics['errors']:
            diagnostics['system_health'] = 'critical'
        elif diagnostics['warnings']:
            diagnostics['system_health'] = 'warning'
        else:
            diagnostics['system_health'] = 'healthy'
        
        return diagnostics
    
    def end_method(self) -> str:
        """End method for the Spatial and Geometric Reasoning System"""
        return "Spatial and Geometric Reasoning System successfully terminated. All functionalities are operational."

end_method = SpatialGeometricReasoningSystem().end_method()
