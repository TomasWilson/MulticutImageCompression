#pragma once
#include "codec.h"
#include "multicut.h"
#include "arithmetic.h"

struct MeanCodec : PartitionCodecBase {

    MeanCodec() {

    }

    const cv::Mat* img;
    const std::vector<PartitionData>* partitions;
    std::vector<cv::Vec3f> key_to_mean_color;
    std::vector<float> key_to_error;

    virtual void initialize(const std::vector<PartitionData>* partitions, const cv::Mat* img) {
        (this->partitions) = partitions;
        (this->img = img);
        key_to_mean_color.clear();
        key_to_error.clear();
        key_to_mean_color.resize(partitions->size());
        key_to_error.resize(partitions->size());
    }

    virtual void NOINLINE write_encoding(BitStream& bs) {
        for(partition_key pk = 0; pk < partitions->size(); pk++) {
            cv::Vec3b mean_color = init_mean_color(pk);
            bs.append(mean_color[0], 8);
            bs.append(mean_color[1], 8);
            bs.append(mean_color[2], 8);
        }
    }

    virtual EncodingResult NOINLINE test_encoding(partition_key pk) {
        return {24, key_to_error.at(pk)};
    }

    virtual EncodingResult NOINLINE test_join_encoding(partition_key pk1, partition_key pk2) {
        const cv::Vec3f& old_mean1 = key_to_mean_color.at(pk1);
        const cv::Vec3f& old_mean2 = key_to_mean_color.at(pk2);

        float n = partitions->at(pk1).points.size() + partitions->at(pk2).points.size();
        float f1 = partitions->at(pk1).points.size() / n;
        float f2 = partitions->at(pk2).points.size() / n;
        cv::Vec3f new_mean = f1 * old_mean1 + f2 * old_mean2;

        // see proof.md
        float   lb_mean_1 = (old_mean1[0] - new_mean[0]) * (old_mean1[0] - new_mean[0]);
                lb_mean_1 += (old_mean1[1] - new_mean[1]) * (old_mean1[1] - new_mean[1]);
                lb_mean_1 += (old_mean1[2] - new_mean[2]) * (old_mean1[2] - new_mean[2]);
                lb_mean_1 *= partitions->at(pk1).points.size();

        float   lb_mean_2 = (old_mean2[0] - new_mean[0]) * (old_mean2[0] - new_mean[0]);
                lb_mean_2 += (old_mean2[1] - new_mean[1]) * (old_mean2[1] - new_mean[1]);
                lb_mean_2 += (old_mean2[2] - new_mean[2]) * (old_mean2[2] - new_mean[2]);
                lb_mean_2 *= partitions->at(pk2).points.size();

        float new_err = key_to_error.at(pk1) + key_to_error.at(pk2) + lb_mean_1 + lb_mean_2;
        return {24, new_err};
    }

    virtual void decode(BitStreamReader& bs, cv::Mat& out_img) {

        for(partition_key pk = 0; pk < partitions->size(); pk++) {
            int b = bs.read8u();
            int g = bs.read8u();
            int r = bs.read8u();
            cv::Vec3b color(b, g, r);

            for(const cv::Point2i& p : partitions->at(pk).points) {
                out_img.at<cv::Vec3b>(p) = color;
            }
        }

    }

    // informs the codec that partition pk is about to be created
    virtual void notify_init(partition_key pk) {
        key_to_mean_color.at(pk) = init_mean_color(pk);
        key_to_error.at(pk) = init_error(pk);
    }

    // inform the codec, that pk1 and pk2 are about to be joined
    // this updates the mean color and error for the partition
    virtual void notify_join(partition_key pk1, partition_key pk2) {

        const cv::Vec3f& old_mean1 = key_to_mean_color.at(pk1);
        const cv::Vec3f& old_mean2 = key_to_mean_color.at(pk2);

        float n = partitions->at(pk1).points.size() + partitions->at(pk2).points.size();
        float f1 = partitions->at(pk1).points.size() / n;
        float f2 = partitions->at(pk2).points.size() / n;
        cv::Vec3f new_mean = f1 * old_mean1 + f2 * old_mean2;
        key_to_mean_color.at(pk1) = new_mean;
        key_to_mean_color.at(pk2) = new_mean;

        // see proof.md
        float   lb_mean_1 = (old_mean1[0] - new_mean[0]) * (old_mean1[0] - new_mean[0]);
                lb_mean_1 += (old_mean1[1] - new_mean[1]) * (old_mean1[1] - new_mean[1]);
                lb_mean_1 += (old_mean1[2] - new_mean[2]) * (old_mean1[2] - new_mean[2]);
                lb_mean_1 *= partitions->at(pk1).points.size();

        float   lb_mean_2 = (old_mean2[0] - new_mean[0]) * (old_mean2[0] - new_mean[0]);
                lb_mean_2 += (old_mean2[1] - new_mean[1]) * (old_mean2[1] - new_mean[1]);
                lb_mean_2 += (old_mean2[2] - new_mean[2]) * (old_mean2[2] - new_mean[2]);
                lb_mean_2 *= partitions->at(pk2).points.size();

        float new_err = key_to_error.at(pk1) + key_to_error.at(pk2) + lb_mean_1 + lb_mean_2;
        key_to_error.at(pk1) = new_err;
        key_to_error.at(pk2) = new_err;

    }

    virtual std::unique_ptr<PartitionCodecBase> clone() const {
        return std::make_unique<MeanCodec>(*this);
    }

protected:

    virtual float NOINLINE init_error(partition_key pk) {

        float error = 0;
        const cv::Vec3f& color = key_to_mean_color.at(pk);

        for(const cv::Point2i& p : partitions->at(pk).points) {
            const cv::Vec3b& p_color = img->at<cv::Vec3b>(p);
            float a = float(color[0]) - float(p_color[0]);
            float b = float(color[1]) - float(p_color[1]);
            float c = float(color[2]) - float(p_color[2]);
            error += a*a + b*b + c*c;
        }

        return error;
    }

    virtual cv::Vec3f NOINLINE init_mean_color(partition_key pk) {
        cv::Vec3d total_color(0, 0, 0);

        for(const cv::Point2i& p : partitions->at(pk).points) {
            total_color += img->at<cv::Vec3b>(p);
        }

        total_color /= (int32_t)partitions->at(pk).points.size();
        return total_color;
    }

};


struct DifferentialMeanCodec : public MeanCodec {

    virtual void NOINLINE write_encoding(BitStream& bs) {

        std::vector<int> db; // distance to previous partition b channel 
        std::vector<int> dg; // distance from b to g channel
        std::vector<int> dr; // distance from g to r channel

        cv::Vec3b first = init_mean_color(0);
        db.push_back(first[0]);
        dg.push_back(first[1] - first[0]);
        dr.push_back(first[2] - first[1]);

        int last_b = first[0];

        for(partition_key pk = 1; pk < partitions->size(); pk++) {
            cv::Vec3b cur = init_mean_color(pk);
            db.push_back(cur[0] - last_b);
            last_b = cur[0];
            dg.push_back(cur[1] - cur[0]);
            dr.push_back(cur[2] - cur[1]);
        }

        encode_sequence<-255, 255>(db, bs, 16);
        encode_sequence<-255, 255>(dg, bs, 16);
        encode_sequence<-255, 255>(dr, bs, 16);

    }

    virtual void decode(BitStreamReader& reader, cv::Mat& out_img) {

        std::vector<int> db = decode_sequence<-255, 255>(reader, 16);
        std::vector<int> dg = decode_sequence<-255, 255>(reader, 16);
        std::vector<int> dr = decode_sequence<-255, 255>(reader, 16);

        std::vector<cv::Vec3b> colors;

        colors.emplace_back(db[0], dg[0] + db[0], dr[0] + dg[0] + db[0]);

        for(int i = 1; i < partitions->size(); i++) {
            int b = db[i] + colors[colors.size() - 1][0];
            colors.emplace_back(
                b,
                dg[i] + b,
                dr[i] + dg[i] + b
            );
        }

        for(partition_key pk = 0; pk < partitions->size(); pk++) {
            for(const cv::Point2i& p : partitions->at(pk).points) {
                out_img.at<cv::Vec3b>(p) = colors[pk];
            }
        }

    }

    virtual std::unique_ptr<PartitionCodecBase> clone() const {
        return std::make_unique<DifferentialMeanCodec>(*this);
    }

};