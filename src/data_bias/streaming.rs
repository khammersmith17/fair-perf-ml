use super::PreTrainingDistribution;
use crate::data_handler::BiasSegmentationCriteria;
use crate::reporting::DataBiasRuntimeReport;
use crate::zip_iters;
use std::collections::HashMap;
/// Container for long running processes and accumulating runtime data. As opposed to point in time
/// utilities provided in the crate.

// TODO:
// baseline computation on init
// accumulate labels based on segmentation criteria
// compute metrics on every data push
// determine best way to have data precomputed
pub struct StreamingDataBias<G, F>
where
    G: PartialEq + PartialOrd,
    F: PartialEq + PartialOrd,
{
    feature_seg_criteria: BiasSegmentationCriteria<F>,
    gt_seg_criteria: BiasSegmentationCriteria<G>,
    rt: DbStreamingGroup,
    bl: DbStreamingGroup,
}

#[derive(Default)]
struct DbStreamingGroup {
    dist_a: PreTrainingDistribution,
    dist_d: PreTrainingDistribution,
}

impl DbStreamingGroup {
    fn construct_bl_dists<F, G>(
        feature_seg: &BiasSegmentationCriteria<F>,
        feature_data: &[F],
        gt_seg: &BiasSegmentationCriteria<G>,
        gt_data: &[G],
    ) -> DbStreamingGroup
    where
        F: PartialEq + PartialOrd,
        G: PartialEq + PartialOrd,
    {
        // returns (facet a, facet d)
        let mut len_a = 0_u64;
        let mut positive_a = 0_u64;

        let mut len_d = 0_u64;
        let mut positive_d = 0_u64;

        for (f, g) in zip_iters!(feature_data, gt_data) {
            let is_favored = feature_seg.label(f);
            let is_positive = gt_seg.label(g);

            len_a += is_favored as u64;
            positive_a += (is_favored && is_positive) as u64;
            len_d += !is_favored as u64;
            positive_d += (!is_favored && is_positive) as u64;
        }

        let dist_a = PreTrainingDistribution {
            len: len_a,
            positive: positive_a,
        };
        let dist_d = PreTrainingDistribution {
            len: len_d,
            positive: positive_d,
        };

        DbStreamingGroup { dist_a, dist_d }
    }
}

impl<G, F> StreamingDataBias<G, F>
where
    F: PartialOrd + PartialEq,
    G: PartialOrd + PartialEq,
{
    pub fn new_with_baseline(
        feature_seg_criteria: BiasSegmentationCriteria<F>,
        gt_seg_criteria: BiasSegmentationCriteria<G>,
        feature_data: &[F],
        gt_data: &[G],
    ) -> StreamingDataBias<G, F> {
        let bl = DbStreamingGroup::construct_bl_dists(
            &feature_seg_criteria,
            feature_data,
            &gt_seg_criteria,
            gt_data,
        );
        let rt = DbStreamingGroup::default();

        StreamingDataBias {
            feature_seg_criteria,
            gt_seg_criteria,
            bl,
            rt,
        }
    }

    pub fn push_data(&mut self, feature: &[F], gt: &[G]) -> DataBiasRuntimeReport {
        todo!()
    }
}

// the idea here is to register many different features
// so it will probably look some thing like a table of <feature name>: <StreamingDataBias>
// figure out how out how to make it store arbitrary types
// I guess they each streaming agent will then need to be boxed
pub struct StreamingDataBiasManager {}
