// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"

#include <magic_enum/magic_enum.hpp>

namespace VideoCore {

vk::ImageViewType ConvertImageViewType(AmdGpu::ImageType type) {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return vk::ImageViewType::e1D;
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageViewType::e1DArray;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return vk::ImageViewType::e2D;
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageViewType::e2DArray;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageViewType::e3D;
    default:
        UNREACHABLE();
    }
}

bool IsViewTypeCompatible(AmdGpu::ImageType view_type, AmdGpu::ImageType image_type) {
    switch (view_type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return image_type == AmdGpu::ImageType::Color1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DMsaaArray:
        return image_type == AmdGpu::ImageType::Color2D || image_type == AmdGpu::ImageType::Color3D;
    case AmdGpu::ImageType::Color3D:
        return image_type == AmdGpu::ImageType::Color3D;
    default:
        UNREACHABLE();
    }
}

ImageViewInfo::ImageViewInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept
    : is_storage{desc.is_written} {
    const auto dfmt = image.GetDataFmt();
    auto nfmt = image.GetNumberFmt();
    if (is_storage && nfmt == AmdGpu::NumberFormat::Srgb) {
        nfmt = AmdGpu::NumberFormat::Unorm;
    }
    format = Vulkan::LiverpoolToVK::SurfaceFormat(dfmt, nfmt);
    if (desc.is_depth) {
        format = Vulkan::LiverpoolToVK::PromoteFormatToDepth(format);
    }

    range.base.level = image.base_level;
    range.base.layer = image.base_array;
    range.extent.levels = image.NumViewLevels(desc.is_array);
    range.extent.layers = image.NumViewLayers(desc.is_array);
    type = image.GetViewType(desc.is_array);

    if (!is_storage) {
        mapping = Vulkan::LiverpoolToVK::ComponentMapping(image.DstSelect());
    }
}

ImageViewInfo::ImageViewInfo(const AmdGpu::ColorBuffer& col_buffer) noexcept {
    range.base.layer = col_buffer.BaseSlice();
    range.extent.layers = col_buffer.NumSlices() - range.base.layer;
    type = range.extent.layers > 1 ? AmdGpu::ImageType::Color2DArray : AmdGpu::ImageType::Color2D;
    format =
        Vulkan::LiverpoolToVK::SurfaceFormat(col_buffer.GetDataFmt(), col_buffer.GetNumberFmt());
}

ImageViewInfo::ImageViewInfo(const AmdGpu::DepthBuffer& depth_buffer, AmdGpu::DepthView view,
                             AmdGpu::DepthControl ctl) {
    format = Vulkan::LiverpoolToVK::DepthFormat(depth_buffer.z_info.format,
                                                depth_buffer.stencil_info.format);
    is_storage = ctl.depth_write_enable;
    range.base.layer = view.slice_start;
    range.extent.layers = view.NumSlices() - range.base.layer;
    type = range.extent.layers > 1 ? AmdGpu::ImageType::Color2DArray : AmdGpu::ImageType::Color2D;
}

ImageView::ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info_,
                     const Image& image)
    : info{info_} {
    vk::ImageViewUsageCreateInfo usage_ci{.usage = image.usage_flags};
    if (!info.is_storage) {
        usage_ci.usage &= ~vk::ImageUsageFlagBits::eStorage;
    }
    // When sampling D32/D16 texture from shader, the T# specifies R32/R16 format so adjust it.
    vk::Format format = info.format;
    vk::ImageAspectFlags aspect = image.aspect_mask;
    if (image.aspect_mask & vk::ImageAspectFlagBits::eDepth &&
        Vulkan::LiverpoolToVK::IsFormatDepthCompatible(format)) {
        format = image.info.pixel_format;
        aspect = vk::ImageAspectFlagBits::eDepth;
    }
    if (image.aspect_mask & vk::ImageAspectFlagBits::eStencil &&
        Vulkan::LiverpoolToVK::IsFormatStencilCompatible(format)) {
        format = image.info.pixel_format;
        aspect = vk::ImageAspectFlagBits::eStencil;
    }

    // Get the requested view type
    vk::ImageViewType requested_view_type = ConvertImageViewType(info.type);
    vk::ImageViewType final_view_type = requested_view_type;

    // Check compatibility and fix if needed
    bool view_type_fixed = false;
    if (!IsViewTypeCompatible(info.type, image.info.type)) {
        LOG_ERROR(Render_Vulkan, "Image view type {} is incompatible with image type {}",
                  magic_enum::enum_name(info.type), magic_enum::enum_name(image.info.type));

        // Fix the view type based on the actual image type
        switch (image.info.type) {
        case AmdGpu::ImageType::Color1D:
            final_view_type = vk::ImageViewType::e1D;
            break;
        case AmdGpu::ImageType::Color1DArray:
            final_view_type = vk::ImageViewType::e1DArray;
            break;
        case AmdGpu::ImageType::Color2D:
        case AmdGpu::ImageType::Color2DMsaa:
            final_view_type = vk::ImageViewType::e2D;
            break;
        case AmdGpu::ImageType::Color2DArray:
        case AmdGpu::ImageType::Color2DMsaaArray:
            final_view_type = vk::ImageViewType::e2DArray;
            break;
        case AmdGpu::ImageType::Color3D:
            final_view_type = vk::ImageViewType::e3D;
            break;
        case AmdGpu::ImageType::Cube:
            final_view_type = vk::ImageViewType::eCube;
            break;
        default:
            LOG_CRITICAL(Render_Vulkan, "Unhandled image type {} for view compatibility fix",
                         static_cast<int>(image.info.type));
            // Fall back to requested type and hope for the best
            final_view_type = requested_view_type;
            break;
        }

        LOG_WARNING(Render_Vulkan, "  -> Forcing view type from {} to {}",
                    vk::to_string(requested_view_type), vk::to_string(final_view_type));
        view_type_fixed = true;
    }

    // Validate and clamp subresource range
    u32 max_layers = 1;
    u32 max_levels = image.info.resources.levels;

    // Determine max layers based on image type
    switch (image.info.type) {
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
        max_layers = image.info.resources.layers;
        break;
    case AmdGpu::ImageType::Cube:
        max_layers = 6; // Cube maps always have 6 layers
        break;
   // case AmdGpu::ImageType::CubeArray:
     //   max_layers = image.info.resources.layers; // Cube array has multiple cubes (each 6 layers)
     //   break;
    default:
        max_layers = 1;
        break;
    }

    // Clamp the subresource range to valid values
    u32 base_layer = std::min(info.range.base.layer, max_layers - 1);
    u32 layer_count = std::min(info.range.extent.layers, max_layers - base_layer);
    u32 base_level = std::min(info.range.base.level, max_levels - 1);
    u32 level_count = std::min(info.range.extent.levels, max_levels - base_level);

    // Log if clamping occurred
    if (view_type_fixed || base_layer != info.range.base.layer ||
        layer_count != info.range.extent.layers || base_level != info.range.base.level ||
        level_count != info.range.extent.levels) {

        if (base_layer != info.range.base.layer || layer_count != info.range.extent.layers) {
            LOG_WARNING(Render_Vulkan, "Clamped layer range: {}+{} -> {}+{} (max layers: {})",
                        info.range.base.layer, info.range.extent.layers, base_layer, layer_count,
                        max_layers);
        }

        if (base_level != info.range.base.level || level_count != info.range.extent.levels) {
            LOG_WARNING(Render_Vulkan, "Clamped mip level range: {}+{} -> {}+{} (max levels: {})",
                        info.range.base.level, info.range.extent.levels, base_level, level_count,
                        max_levels);
        }
    }

    const vk::ImageViewCreateInfo image_view_ci = {
        .pNext = &usage_ci,
        .image = image.GetImage(),
        .viewType = final_view_type,
        .format = instance.GetSupportedFormat(format, image.format_features),
        .components = info.mapping,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = base_level,
            .levelCount = level_count,
            .baseArrayLayer = base_layer,
            .layerCount = layer_count,
        },
    };

    auto [view_result, view] = instance.GetDevice().createImageViewUnique(image_view_ci);
    ASSERT_MSG(view_result == vk::Result::eSuccess, "Failed to create image view: {}",
               vk::to_string(view_result));
    image_view = std::move(view);

    const auto view_aspect = aspect & vk::ImageAspectFlagBits::eDepth     ? "Depth"
                             : aspect & vk::ImageAspectFlagBits::eStencil ? "Stencil"
                                                                          : "Color";
    Vulkan::SetObjectName(instance.GetDevice(), *image_view,
                          "ImageView {}x{}x{} {:#x}:{:#x} {}:{} {}:{} ({})", image.info.size.width,
                          image.info.size.height, image.info.size.depth, image.info.guest_address,
                          image.info.guest_size, base_level, base_level + level_count - 1,
                          base_layer, base_layer + layer_count - 1, view_aspect);
}
ImageView::~ImageView() = default;

} // namespace VideoCore
