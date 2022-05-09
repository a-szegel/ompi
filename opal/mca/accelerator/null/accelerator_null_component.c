/*
 * Copyright (c) 2014-2015 Intel, Inc.  All rights reserved.
 * Copyright (c) 2014      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2017-2022 Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"
#include "opal/constants.h"
#include "accelerator_null_component.h"

/*
 * Public string showing the accelerator null component version number
 */
const char *opal_accelerator_null_component_version_string
    = "OPAL null accelerator MCA component version " OPAL_VERSION;

/*
 * Component API functions
 */
static int accelerator_null_open(void);
static int accelerator_null_close(void);
static int accelerator_null_component_register(void);
static opal_accelerator_base_module_t* accelerator_null_init(void);
static void accelerator_null_finalize(opal_accelerator_base_module_t* module);

/* Accelerator API's */
static int accelerator_null_check_addr(const void *addr, int *dev_id, uint64_t *flags);

static int accelerator_null_create_stream(int dev_id, opal_accelerator_stream_t **stream);
static int accelerator_null_destroy_stream(opal_accelerator_stream_t *stream);
static int accelerator_null_synchronize_stream(opal_accelerator_stream_t *stream);
static int accelerator_null_create_event(opal_accelerator_event_t *event);
static int accelerator_null_destroy_event(opal_accelerator_event_t *event);
static int accelerator_null_record_event(opal_accelerator_event_t *event, opal_accelerator_stream_t *stream);
static int accelerator_null_query_event(opal_accelerator_event_t *event);
static int accelerator_null_synchronize_event(opal_accelerator_event_t *event);

static int accelerator_null_memcpy_async(int dev_id, void *dest, const void *src, size_t size,
                                         opal_accelerator_stream_t *stream, opal_accelerator_transfer_type_t type);
static int accelerator_null_memcpy(int dev_id, void *dest, const void *src,
                                   size_t size, opal_accelerator_transfer_type_t type);
static int accelerator_null_vector_memcpy(int dev_id, void *dest, size_t dpitch,
                                          const void *src, size_t spitch,
                                          size_t width, size_t height,
                                          opal_accelerator_transfer_type_t type);
static int accelerator_null_memmove(int dev_id, void *dest, const void *src, size_t size,
                                    opal_accelerator_transfer_type_t type);

static int accelerator_null_malloc(void **ptr, size_t size);
static int accelerator_null_free(void *ptr);
static int accelerator_null_get_address_range(const void *ptr, void **base, size_t *size);

static bool accelerator_null_is_ipc_enabled(void);
static size_t accelerator_null_get_handle_size(void);
static int accelerator_null_get_handle(void *dev_ptr, void **handle);
static int accelerator_null_open_handle(void *handle, int dev_id,  void **dev_ptr);
static int accelerator_null_close_handle(void *dev_ptr);

static int accelerator_null_host_register(void *ptr, size_t size);
static int accelerator_null_host_unregister(void *ptr);

static int accelerator_null_get_device(int *dev_id);
static int accelerator_null_device_can_access_peer( int *access, int dev1, int dev2);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

opal_accelerator_null_component_t mca_accelerator_null_component = {{

    /* First, the mca_component_t struct containing meta information
     * about the component itself */

    .base_version =
        {
            /* Indicate that we are a accelerator v1.1.0 component (which also
             * implies a specific MCA version) */

            OPAL_ACCELERATOR_BASE_VERSION_1_0_0,

            /* Component name and version */

            .mca_component_name = "null",
            MCA_BASE_MAKE_VERSION(component, OPAL_MAJOR_VERSION, OPAL_MINOR_VERSION,
                                  OPAL_RELEASE_VERSION),

            /* Component open and close functions */

            .mca_open_component = accelerator_null_open,
            .mca_close_component = accelerator_null_close,
            .mca_register_component_params = accelerator_null_component_register,

        },
    /* Next the MCA v1.0.0 component meta data */
    .base_data =
        { /* The component is checkpoint ready */
         MCA_BASE_METADATA_PARAM_CHECKPOINT},
    .accelerator_init = accelerator_null_init,
    .accelerator_finalize = accelerator_null_finalize,
}};

opal_accelerator_base_module_t opal_accelerator_null_module =
{
    accelerator_null_check_addr,

    accelerator_null_create_stream,
    accelerator_null_destroy_stream,
    accelerator_null_synchronize_stream,

    accelerator_null_create_event,
    accelerator_null_destroy_event,
    accelerator_null_record_event,
    accelerator_null_query_event,
    accelerator_null_synchronize_event,

    accelerator_null_memcpy_async,
    accelerator_null_memcpy,
    accelerator_null_vector_memcpy,
    accelerator_null_memmove,
    accelerator_null_malloc,
    accelerator_null_free,
    accelerator_null_get_address_range,

    accelerator_null_is_ipc_enabled,
    accelerator_null_get_handle_size,
    accelerator_null_get_handle,
    accelerator_null_open_handle,
    accelerator_null_close_handle,

    accelerator_null_host_register,
    accelerator_null_host_unregister,

    accelerator_null_get_device,
    accelerator_null_device_can_access_peer
};

static int accelerator_null_open(void)
{
    return OPAL_SUCCESS;
}

static int accelerator_null_close(void)
{
    return OPAL_SUCCESS;
}

static int accelerator_null_component_register(void)
{
    return OPAL_SUCCESS;
}

static opal_accelerator_base_module_t* accelerator_null_init(void)
{
    return &opal_accelerator_null_module;
}

static void accelerator_null_finalize(opal_accelerator_base_module_t* module)
{
    return;
}

/* Accelerator API's Implementation */
static int accelerator_null_check_addr(const void *addr, int *dev_id, uint64_t *flags)
{
    /* Always return that the pointer belongs to the host */
    return 0;
}


static int accelerator_null_create_stream(int dev_id, opal_accelerator_stream_t **stream)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_destroy_stream(opal_accelerator_stream_t *stream)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_synchronize_stream(opal_accelerator_stream_t *stream)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_create_event(opal_accelerator_event_t *event)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_destroy_event(opal_accelerator_event_t *event)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_record_event(opal_accelerator_event_t *event, opal_accelerator_stream_t *stream)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_query_event(opal_accelerator_event_t *event)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_synchronize_event(opal_accelerator_event_t *event)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_memcpy_async(int dev_id, void *dest, const void *src, size_t size,
                                  opal_accelerator_stream_t *stream, opal_accelerator_transfer_type_t type)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_memcpy(int dev_id, void *dest, const void *src,
                            size_t size, opal_accelerator_transfer_type_t type)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_vector_memcpy(int dev_id, void *dest, size_t dpitch,
                                   const void *src, size_t spitch,
                                   size_t width, size_t height,
                                   opal_accelerator_transfer_type_t type)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_memmove(int dev_id, void *dest, const void *src, size_t size,
                             opal_accelerator_transfer_type_t type)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_malloc(void **ptr, size_t size)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_free(void *ptr)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_get_address_range(const void *ptr, void **base,
                                              size_t *size)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static bool accelerator_null_is_ipc_enabled(void)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static size_t accelerator_null_get_handle_size(void)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_get_handle(void *dev_ptr, void **handle)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_open_handle(void *handle, int dev_id,  void **dev_ptr)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_close_handle(void *dev_ptr)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_host_register(void *ptr, size_t size)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_host_unregister(void *ptr)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_get_device(int *dev_id)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}

static int accelerator_null_device_can_access_peer( int *access, int dev1, int dev2)
{
    return OPAL_ERR_NOT_IMPLEMENTED;
}
