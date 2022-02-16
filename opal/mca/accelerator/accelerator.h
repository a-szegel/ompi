/*
 * Copyright (c) 2014-2021 Intel, Inc. All rights reserved.
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

#ifndef OPAL_ACCELERATOR_H
#define OPAL_ACCELERATOR_H

#include "opal/class/opal_object.h"
#include "opal/mca/mca.h"

BEGIN_C_DECLS

/**
 * Accelerator flags
 */
/* Unified memory buffers */
#define MCA_ACCELERATOR_FLAGS_UNIFIED_MEMORY 0x00000001

/**
 * Transfer types.
 * UNSPEC - Not specified
 * HTOH - Host to Host
 * HTOD - Host to Device
 * DTOH - Device to Host
 * DTOD - Device to Device
 */
typedef enum {
    MCA_ACCELERATOR_TRANSFER_UNSPEC = 0,
    MCA_ACCELERATOR_TRANSFER_HTOH,
    MCA_ACCELERATOR_TRANSFER_HTOD,
    MCA_ACCELERATOR_TRANSFER_DTOH,
    MCA_ACCELERATOR_TRANSFER_DTOD,
} opal_accelerator_transfer_type_t;

struct opal_accelerator_stream_t {
    opal_object_t super;
    /* Stream object */
    void *stream;
};
typedef struct opal_accelerator_stream_t opal_accelerator_stream_t;
OBJ_CLASS_DECLARATION(opal_accelerator_stream_t);

struct opal_accelerator_event_t {
    opal_object_t super;
    /* Event object */
    void *event;
};
typedef struct opal_accelerator_event_t opal_accelerator_event_t;
OBJ_CLASS_DECLARATION(opal_accelerator_event_t);

/**
 * Check whether a pointer belongs to an accelerator or not.
 * interfaces
 *
 * @param[IN] addr           Pointer to check
 * @param[OUT] dev_id        Returns the device id against which the memory was allocated
 * @param[OUT] flags         May be set to indicate additional information
 *                           about the corresponding pointer such
 *                           as whether it belongs to unified memory.
 *
 * @retval <0                An error has occurred.
 * @retval 0                 The buffer does not belong to a managed buffer
 *                           in device memory.
 * @retval >0                The buffer belongs to a managed buffer in
 *                           device memory.
 */
typedef int (*opal_accelerator_base_module_check_addr_fn_t)(
    const void *addr, int *dev_id, uint64_t *flags);

/**
 * Creates a stream for asynchonous operations. This will also
 * allocate the stream object.
 *
 * @param[IN] dev_id         Associated device for the stream
 * @param[OUT] stream        Returned stream object
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_create_stream_fn_t)(
    int dev_id, opal_accelerator_stream_t **stream);

/**
 * Destroys a stream created by opal_accelerator_base_module_create_stream_fn_t().
 * This will also free the stream object.
 *
 * @param[IN] stream         Stream to destroy
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_destroy_stream_fn_t)(
    opal_accelerator_stream_t *stream);

/**
 * Waits until all events on the stream are completed.
 *
 * @param[IN] stream         Stream to wait for
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_synchronize_stream_fn_t)(
    opal_accelerator_stream_t *stream);

/**
 * Creates an event. You must allocate memory for event prior to
 * this call.
 *
 * @param[IN] event          Event to create
 *
 * @return                   OPAL_SUCCESS or error status on failure.
 */
typedef int (*opal_accelerator_base_module_create_event_fn_t)(
    opal_accelerator_event_t *event);

/**
 * Destroys an event.
 *
 * @param[IN] event          Event to destroy
 *
 * @return                   OPAL_SUCCESS or error status on failure.
 */
typedef int (*opal_accelerator_base_module_destroy_event_fn_t)(
    opal_accelerator_event_t *event);

/**
 * Records an event on a stream.
 *
 * @param[IN] event          Event to record
 * @param[IN] stream         Stream to record event for
 *
 * @return                   OPAL_SUCCESS or error status on failure.
 */
typedef int (*opal_accelerator_base_module_record_event_fn_t)(
    opal_accelerator_event_t *event, opal_accelerator_stream_t *stream);

/**
 * Queries an event's status.
 *
 * @param[IN] event          Event to query
 *
 * @return                   OPAL_SUCCESS on event completion, OPAL_ERROR on error,
 *                           or OPAL_ERR_RESOURCE_BUSY if any work is incomplete.
 */
typedef int (*opal_accelerator_base_module_query_event_fn_t)(
    opal_accelerator_event_t *event);

/**
 * Waits for an event to complete.
 *
 * @param[IN] event          Event to wait for
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_synchronize_event_fn_t)(
    opal_accelerator_event_t *event);

/**
 * Copies memory asynchronously from src to dest. Memory of dest and src
 * may not overlap. Optionally can specify the transfer type to
 * avoid pointer detection for performance.
 *
 * @param[IN] dev_id         Associated device to copy to/from
 * @param[IN] dest           Destination to copy memory to
 * @param[IN] src            Source to copy memory from
 * @param[IN] size           Size of memory to copy
 * @param[IN] stream         Stream to perform asynchronous copy on
 * @param[IN] type           Transfer type field for performance
 *                           Can be set to MCA_ACCELERATOR_TRANSFER_UNSPEC
 *                           if transfer type is not specified.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_memcpy_async_fn_t)(
    int dev_id, void *dest, const void *src, size_t size,
    opal_accelerator_stream_t *stream, opal_accelerator_transfer_type_t type);

/**
 * Copies memory synchronously from src to dest. Memory of dest and src
 * may not overlap. Optionally can specify the transfer type to
 * avoid pointer detection for performance.
 *
 * @param[IN] dev_id         Associated device to copy to/from
 * @param[IN] dest           Destination to copy memory to
 * @param[IN] src            Source to copy memory from
 * @param[IN] size           Size of memory to copy
 * @param[IN] type           Transfer type field for performance
 *                           Can be set to MCA_ACCELERATOR_TRANSFER_UNSPEC
 *                           if transfer type is not specified.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_memcpy_fn_t)(
    int dev_id, void *dest, const void *src, size_t size,
    opal_accelerator_transfer_type_t type);

/**
 * Copies a matrix of memory (height rows of width bytes) synchronously
 * from src to dest. Memory of dest and src may not overlap. Optionally
 * can specify the transfer type to avoid pointer detection for
 * performance.
 *
 * @param[IN] dev_id         Associated device to copy to/from
 * @param[IN] dest           Destination to copy memory to
 * @param[IN] dpitch         Pitch of destination memory
 * @param[IN] src            Source to copy memory from
 * @param[IN] spitch         Pitch of source memory
 * @param[IN] width          Width of matrix transfer (columns in bytes)
 * @param[IN] height         Height of matrix transfer (rows)
 * @param[IN] type           Transfer type field for performance
 *                           Can be set to MCA_ACCELERATOR_TRANSFER_UNSPEC
 *                           if transfer type is not specified.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_vector_memcpy_fn_t)(
    int dev_id, void *dest, size_t dpitch, const void *src, size_t spitch,
    size_t width, size_t height, opal_accelerator_transfer_type_t type);

/**
 * Copies memory synchronously from src to dest. Memory of dest and src
 * may overlap. Optionally can specify the transfer type to
 * avoid pointer detection for performance.
 *
 * @param[IN] dev_id         Associated device to copy to/from
 * @param[IN] dest           Destination to copy memory to
 * @param[IN] src            Source to copy memory from
 * @param[IN] size           Size of memory to copy
 * @param[IN] type           Transfer type field for performance
 *                           Can be set to MCA_ACCELERATOR_TRANSFER_UNSPEC
 *                           if transfer type is not specified.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_memmove_fn_t)(
    int dev_id, void *dest, const void *src, size_t size,
    opal_accelerator_transfer_type_t type);

/**
 * Allocates size bytes memory from the device and sets ptr to the
 * pointer of the allocated memory. The memory is not initialized.
 * If size is 0, then the function will return 0 and ptr will not be set.
 * If multiple devices are present, a component can decide which
 * device to allocate memory from.
 *
 * @param[OUT] ptr           Returns pointer to allocated memory
 * @param[IN] size           Size of memory to allocate
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_malloc_fn_t)(
    void **ptr, size_t size);

/**
 * Frees the memory space pointed to by ptr which has been returned by
 * a previous call to an opal_accelerator_base_module_malloc_fn_t().
 * If the function is called on a ptr that has already been freed,
 * undefined behavior occurs. If ptr is NULL, no operation is performed,
 * and the function returns OPAL_SUCCESS.
 *
 * @param[IN] ptr            Pointer to free
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_free_fn_t)(
    void *ptr);

/**
 * Retrieves the base address and/or size of a memory allocation of the
 * device.
 *
 * @param[IN] ptr            Pointer to device memory to get base/size from
 * @param[OUT] base          Base address of the memory allocation
 * @param[OUT] size          Size of the memory allocation
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_get_address_range_fn_t)(
    const void *ptr, void **base, size_t *size);

/**
 * Queries whether the device supports Inter Process Communication
 * or not. If true, the functions:
 *
 * opal_accelerator_base_module_get_handle_size_fn_t()
 * opal_accelerator_base_module_get_handle_fn_t()
 * opal_accelerator_base_module_open_handle_fn_t()
 * opal_accelerator_base_module_close_handle_fn_t()
 *
 * must be implemented.
 *
 * @return true              IPC supported
 * @return false             IPC not supported
 */
typedef bool (*opal_accelerator_base_module_is_ipc_enabled_fn_t)(void);

/**
 * Returns the size of the Inter Process Communication memory handle
 *
 * @return size_t            Size of the IPC memory handle
 */
typedef size_t (*opal_accelerator_base_module_get_handle_size_fn_t)(void);

/**
 * Gets an interprocess memory handle for an existing device memory allocation.
 *
 * @param[IN] dev_ptr        Base pointer to previously allocated device
 *                           memory
 * @param[OUT] handle        Pointer to user allocated mem handle to return
 *                           the memory handle in.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 *
 */
typedef int (*opal_accelerator_base_module_get_handle_fn_t)(
    void *dev_ptr, void **handle);

/**
 * Opens an interprocess memory handle from another process and returns
 * a device pointer usable in the local process.
 *
 * @param[IN] handle         IPC memory handle from another process
 * @param[IN] dev_id         Device ID associated with the IPC memory handle
 * @param[OUT] dev_ptr       Returned device pointer
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_open_handle_fn_t)(
    void *handle, int dev_id, void **dev_ptr);

/**
 * Closes memory mapped with opal_accelerator_base_module_open_handle_fn_t().
 *
 * @param[IN] dev_ptr            IPC device pointer returned from
 *                               opal_accelerator_base_module_open_handle_fn_t()
 */
typedef int (*opal_accelerator_base_module_close_handle_fn_t)(
    void *dev_ptr);

/**
 * Page-locks the memory range specified by ptr and size
 *
 * @param[IN] ptr            Host pointer to memory to page-lock
 * @param[IN] size           Size in bytes of the address range to page-lock in bytes
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_host_register_fn_t)(
    void *ptr, size_t size);

/**
 * Unregisters a memory range that was registered with
 * opal_accelerator_base_module_host_register_fn_t.
 *
 * @param[IN] ptr            Host pointer to memory to unregister
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_host_unregister_fn_t)(
    void *ptr);

/**
 * Retrieves current device id for a device associated with the local process.
 *
 * @param[OUT] dev_id        ID of the device
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_get_device_fn_t)(
    int *dev_id);

/**
 * Queries if a device may directly access a peer device's memory.
 *
 * @param[OUT] access        Returns 1 if dev1 can directly access memory on dev2
 *                           Returns 0 if dev1 can not directly access memory on dev2
 * @param[IN] dev1           ID of device checking if peer device memory can be accessed
 * @param[IN] dev2           ID of peer device on which the memory allocations
 *                           reside.
 *
 * @return                   OPAL_SUCCESS or error status on failure
 */
typedef int (*opal_accelerator_base_module_device_can_access_peer_fn_t)(
    int *access, int dev1, int dev2);

/*
 * the standard public API data structure
 */
typedef struct {
    /* accelerator function table */
    opal_accelerator_base_module_check_addr_fn_t check_addr;

    opal_accelerator_base_module_create_stream_fn_t create_stream;
    opal_accelerator_base_module_destroy_stream_fn_t destroy_stream;
    opal_accelerator_base_module_synchronize_stream_fn_t synchronize_stream;
    opal_accelerator_base_module_create_event_fn_t create_event;
    opal_accelerator_base_module_destroy_event_fn_t destroy_event;
    opal_accelerator_base_module_record_event_fn_t record_event;
    opal_accelerator_base_module_query_event_fn_t query_event;
    opal_accelerator_base_module_synchronize_event_fn_t synchronize_event;

    opal_accelerator_base_module_memcpy_async_fn_t memcpy_async;
    opal_accelerator_base_module_memcpy_fn_t memcpy;
    opal_accelerator_base_module_vector_memcpy_fn_t vector_memcpy;
    opal_accelerator_base_module_memmove_fn_t memmove;

    opal_accelerator_base_module_malloc_fn_t malloc;
    opal_accelerator_base_module_free_fn_t free;
    opal_accelerator_base_module_get_address_range_fn_t get_address_range;

    opal_accelerator_base_module_is_ipc_enabled_fn_t is_ipc_enabled;
    opal_accelerator_base_module_get_handle_size_fn_t get_handle_size;
    opal_accelerator_base_module_get_handle_fn_t get_handle;
    opal_accelerator_base_module_open_handle_fn_t open_handle;
    opal_accelerator_base_module_close_handle_fn_t close_handle;

    opal_accelerator_base_module_host_register_fn_t host_register;
    opal_accelerator_base_module_host_unregister_fn_t host_unregister;

    opal_accelerator_base_module_get_device_fn_t get_device;
    opal_accelerator_base_module_device_can_access_peer_fn_t device_can_access_peer;
} opal_accelerator_base_module_t;

/**
 * Accelerator component initialization.
 * Called by MCA framework to initialize the component.
 *
 * This should initialize any component level data.
 *
 * This should discover accelerators that are available.
 * We assume that only one accelerator will be present
 * on any given node.
 *
 * @return                   Initialized module or NULL if init failed.
 */
typedef opal_accelerator_base_module_t * (*mca_accelerator_base_component_init_fn_t)(void);

/**
 * Accelerator component finalization
 * Called by MCA framework to finalize the component.
 *
 * This should finalize the given accelerator component.
 * Any component level data should be cleaned up, including
 * any allocated during component_init() and data created
 * during the lifetime of the component, including outstanding
 * modules.
 *
 * @param[IN] module    If the component performed allocation within
 *                      the module, allow the component the to perform
 *                      the required cleanup
 *
 * No return since error will likely be ignored anyway.
 */
typedef void (*mca_accelerator_base_component_fini_fn_t)(opal_accelerator_base_module_t* module);

typedef struct {
    mca_base_component_t base_version;
    mca_base_component_data_t base_data;
    mca_accelerator_base_component_init_fn_t accelerator_init;
    mca_accelerator_base_component_fini_fn_t accelerator_finalize;
} opal_accelerator_base_component_t;

/*
 * Macro for use in components that are of type accelerator
 */
#define OPAL_ACCELERATOR_BASE_VERSION_1_0_0 OPAL_MCA_BASE_VERSION_2_1_0("accelerator", 1, 0, 0)

/* Global structure for accessing accelerator functions */
OPAL_DECLSPEC extern opal_accelerator_base_module_t opal_accelerator;

END_C_DECLS

#endif
