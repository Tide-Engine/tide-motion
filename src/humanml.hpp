#pragma once
#include "motion.hpp"

// HumanML representation arithmetic, independent of model loading.
namespace motion::humanml {
inline constexpr size_t features=263, joints=22;
using Row=std::array<float,features>;
using Positions=std::array<Vec3,joints>;
struct Root { Quat world_to_heading; Vec3 position; };
struct Statistics { Row mean{}, deviation{}; void validate() const; };
std::vector<Row> normalize(std::span<const Row>, const Statistics&, bool inverse=false);
std::vector<Root> roots(std::span<const Row>);
std::vector<Positions> positions(std::span<const Row>);
// Upstream's incoming-edge rotation convention, not glTF local rotations.
std::vector<Positions> rotation_fk(std::span<const Row>, const Positions& offsets);
// RIC positions with 6D edge frames as roll hints, fixed canonical SOMA lengths.
// Missing facial/finger joints retain rest rotations. This is a lossy conversion.
Clip to_soma(std::span<const Row>);
struct Encoded { std::vector<Row> rows; Vec3 origin; float heading=0; };
Encoded from_soma(const Clip&);
// A range is [begin,end) on the original canonical 30 Hz timeline. Endpoints
// within the range must already match; no blending or hidden seam correction.
Clip splice(const Clip& source, const Clip& generated, size_t begin, size_t end,
            float position_tolerance=1e-5f, float rotation_tolerance=1e-5f);
size_t mask_count(size_t tokens, size_t iteration, size_t iterations);
}
